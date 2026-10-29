"""Typed query values with independent dense and gathered native executors.

No stochastic operations, public validation or telemetry mutation occur inside
the checkpointed primitives. Supports are fixed before entering them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..content_hypergraph.readers_v2 import DirectSourceReaderV2
from ..content_hypergraph.spatial_features_v2 import query_source_score_features, query_source_features
from .response import JointNativeSourceFactors


@dataclass(frozen=True)
class TypedRead:
    context: Tensor
    alpha: Tensor
    response: object | None
    messages: Tensor | None
    counts: dict[str, int]


def raw_scores(reader, query, query_states, coordinates, lengths, frame_scale, keys):
    """Original narrow scores, [B,H,Q,N], before measures or support."""
    batch, queries, _ = query_states.shape
    count = coordinates.shape[1]
    if not count or not queries:
        return query_states.new_zeros(batch, reader.heads, queries, count)
    descriptor = query_source_score_features(query.coordinates.to(query_states), coordinates.to(query_states),
                                             frame_scale, lengths.to(query_states))
    q = F.normalize(reader.query_key(reader.query_norm(query_states)).reshape(
        batch, queries, reader.heads, reader.head_width), dim=-1, eps=1e-6)
    k = F.normalize(keys.reshape(batch, count, reader.heads, reader.head_width), dim=-1, eps=1e-6)
    return (torch.einsum("bqhd,bnhd->bhqn", q, k) * reader.temperature[None, :, None, None]
            + reader.geometry_score(descriptor).permute(0, 3, 1, 2))


def _multiplicative_weights(scores, gate, measure, valid, query_valid, reference):
    """Zero-sensitive normalizer: omitted gates keep restoration derivatives."""
    if not scores.shape[-1]:
        return scores.permute(0, 2, 3, 1)
    eligible = valid & (measure > 0)
    allowed = eligible[:, None, None, :] & query_valid[:, None, :, None]
    masked = torch.where(allowed, scores, -torch.inf)
    maximum = masked.amax(-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    exponential = torch.exp(torch.where(allowed, scores - maximum, -torch.inf))
    relative_measure = torch.where(eligible, measure, 0) / reference[:, None].clamp_min(torch.finfo(scores.dtype).tiny)
    weight = exponential * relative_measure[:, None, None, :] * gate[:, None]
    denominator = weight.sum(-1, keepdim=True)
    return (weight / denominator.clamp_min(torch.finfo(weight.dtype).tiny)).permute(0, 2, 3, 1)


def _gamma(delta, scale):
    relative = delta / scale
    values = [relative]
    for frequency in (1., 2., 4.):
        phase = 2 * torch.pi * frequency * relative
        values.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(values, -1)


def _flat_messages(reader, qstates, qcoords, coordinates, lengths, scale, values, indices):
    # Gather original records BEFORE wide Fourier/value features.
    b, q, n = indices.unbind(-1)
    delta = qcoords[b, q] - coordinates[b, n]
    relative = reader.relative_value(_gamma(delta, scale[b]))
    if reader.source_scale_projection is not None:
        relative = relative + reader.source_scale_projection(torch.asinh(delta / lengths[b, n]))
    hidden = F.silu(reader.query_value(qstates)[b, q] + values[b, n] + relative)
    messages = reader.message_output(hidden)
    scalar = (reader.thermal_output(messages).squeeze(-1) if reader.thermal_output is not None
              else messages.new_empty((0,)))
    return messages, scalar


def _dense_messages(reader, qstates, qcoords, coordinates, lengths, scale, values):
    # Deliberately independent rectangular implementation for the same mask.
    descriptor, gamma = query_source_features(qcoords, coordinates, scale, lengths)
    relative = reader.relative_value(gamma)
    if reader.source_scale_projection is not None:
        relative = relative + reader.source_scale_projection(descriptor[..., -reader.coordinate_dim:])
    messages = reader.message_output(F.silu(reader.query_value(qstates)[:, :, None]
                                          + values[:, None] + relative))
    scalar = (reader.thermal_output(messages).squeeze(-1) if reader.thermal_output is not None
              else messages.new_empty((0,)))
    return messages, scalar


def _dense_reduce(reader, qstates, qcoords, coordinates, lengths, scale, values, alpha, gate,
                  return_messages=False):
    messages, scalar = _dense_messages(reader, qstates, qcoords, coordinates, lengths, scale, values)
    batch, queries, count = alpha.shape[:3]
    blocks = messages.reshape(batch, queries, count, reader.heads, reader.head_width)
    context = (blocks * alpha[..., None]).sum(2)
    if reader.thermal_output is not None:
        scalar = scalar * gate
    return context, scalar, messages if return_messages else messages.new_empty((0,))


def _flat_reduce(reader, qstates, qcoords, coordinates, lengths, scale, values, indices, alpha,
                 gate, return_messages=False):
    messages, scalar = _flat_messages(reader, qstates, qcoords, coordinates, lengths, scale, values, indices)
    batch, queries = qstates.shape[:2]
    b, q, n = indices.unbind(-1)
    blocks = messages.reshape(-1, reader.heads, reader.head_width) * alpha[b,q,n,:,None]
    context = messages.new_zeros(batch*queries, reader.heads, reader.head_width).index_add(0,b*queries+q,blocks)
    if reader.thermal_output is not None:
        scalar = scalar * gate[b,q,n]
    return context.reshape(batch, queries, reader.heads, reader.head_width), scalar, messages if return_messages else messages.new_empty((0,))


def read_typed(reader: DirectSourceReaderV2, query, query_states, *, coordinates, lengths, valid,
               measure, reference, frame_scale, keys, values, gate, backend="flat_typed_selected",
               return_messages=False, checkpoint_values=False, query_tile_size=None):
    """Read one original type, preserving its own measure and denominator.

    M ``valid`` means physical presence, including zero-information-measure
    heaters. E callers pass positive-quadrature eligibility. Dense references
    multiply signed response coefficients by the gate without attention.
    """
    if backend not in {"flat_typed_selected", "dense_typed_reference", "dense_ME"}:
        raise ValueError("Unknown joint native value backend.")
    batch, queries, width = query_states.shape
    count = coordinates.shape[1]
    query_valid = (torch.ones((batch, queries), device=query_states.device, dtype=torch.bool)
                   if query.query_mask is None else query.query_mask)
    pair_valid = query_valid[:, :, None] & valid[:, None]
    if gate.shape != pair_valid.shape:
        raise ValueError("Typed gate must preserve [B,Q,N] original record axes.")
    gate = gate.to(query_states.dtype) * pair_valid
    support = (gate.detach() > 0) & pair_valid
    scores = raw_scores(reader, query, query_states, coordinates, lengths, frame_scale, keys)
    alpha = _multiplicative_weights(scores, gate, measure.to(query_states), valid, query_valid,
                                     reference.to(query_states))
    indices = support.nonzero(as_tuple=False)
    dense = backend != "flat_typed_selected"
    scalar_dense = query_states.new_zeros(batch, queries, count) if dense and reader.thermal_output is not None else None
    message_grid = query_states.new_zeros(batch, queries, count, width) if return_messages else None
    context = query_states.new_zeros(batch * queries, reader.heads, reader.head_width)
    coefficients, coefficient_indices = [], []
    tile_size = (queries or 1) if query_tile_size is None else query_tile_size
    if type(tile_size) is not int or tile_size < 1:
        raise ValueError("query_tile_size must be positive.")
    value_rows = 0
    for start in range(0, queries, tile_size):
        stop = min(queries, start + tile_size)
        if dense:
            if not count:
                continue
            args = (query_states[:, start:stop], query.coordinates[:, start:stop].to(query_states),
                    coordinates.to(query_states), lengths.to(query_states), frame_scale, values,
                    alpha[:,start:stop], gate[:,start:stop])
            function = lambda *inputs: _dense_reduce(reader, *inputs, return_messages=return_messages)
            chunk, scalar, messages = (checkpoint(function, *args, use_reentrant=False) if checkpoint_values
                                and torch.is_grad_enabled() else function(*args))
            flat_query = (torch.arange(batch, device=query_states.device)[:, None] * queries
                          + torch.arange(start, stop, device=query_states.device)[None]).flatten()
            context = context.index_add(0, flat_query, chunk.flatten(0, 1))
            if scalar_dense is not None:
                scalar_dense[:, start:stop] = scalar
            if message_grid is not None:
                message_grid[:, start:stop] = messages * support[:, start:stop, :, None]
            value_rows += batch * (stop-start) * count
        else:
            rows = indices[(indices[:, 1] >= start) & (indices[:, 1] < stop)]
            if not rows.shape[0]:
                continue
            local_rows = rows - rows.new_tensor([0,start,0])
            args = (query_states[:,start:stop], query.coordinates[:,start:stop].to(query_states), coordinates.to(query_states),
                    lengths.to(query_states), frame_scale, values, local_rows, alpha[:,start:stop], gate[:,start:stop])
            function = lambda *inputs: _flat_reduce(reader, *inputs, return_messages=return_messages)
            chunk, scalar, messages = (checkpoint(function, *args, use_reentrant=False) if checkpoint_values
                                and torch.is_grad_enabled() else function(*args))
            b, q, n = rows.unbind(-1)
            flat_query = (torch.arange(batch,device=query_states.device)[:,None]*queries
                          + torch.arange(start,stop,device=query_states.device)[None]).flatten()
            context = context.index_add(0,flat_query,chunk.flatten(0,1))
            if reader.thermal_output is not None:
                coefficients.append(scalar)
                coefficient_indices.append(rows)
            if message_grid is not None:
                message_grid[b, q, n] = messages
            value_rows += rows.shape[0]
    response = scalar_dense
    if not dense and reader.thermal_output is not None:
        rows = torch.cat(coefficient_indices) if coefficient_indices else indices.new_empty((0, 3))
        coeff = torch.cat(coefficients) if coefficients else query_states.new_empty((0,))
        b, q, n = rows.unbind(-1)
        response = JointNativeSourceFactors(batch, queries, count, b, q, n, coeff, valid,
            query_states.new_tensor(torch.finfo(query_states.dtype).max), query_mask=query_valid)
    counts = dict(value_rows=value_rows, selected_rows=int(indices.shape[0]),
                  candidate_rows=batch*queries*count, score_rows=batch*queries*count,
                  message_projection_rows=value_rows,
                  scalar_projection_rows=value_rows if reader.thermal_output is not None else 0,
                  fourier_entries=value_rows*7*reader.coordinate_dim, hidden_entries=value_rows*width,
                  padded_rows=0, invalid_candidate_rows_available=0,
                  checkpoint_forward_rows=value_rows if checkpoint_values and torch.is_grad_enabled() else 0,
                  checkpoint_recomputation_rows_available=0,
                  dense_value_rows=value_rows if dense else 0)
    return TypedRead(context.reshape(batch, queries, width), alpha, response, message_grid, counts)


__all__ = ["TypedRead", "raw_scores", "read_typed"]
