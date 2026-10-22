"""Continuous typed readers with historical and reduced-projection executors."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..content_hypergraph.spatial_features_v2 import query_source_features
from ..content_hypergraph.source_read_execution import (
    compose_scalar_projection,
    reduce_headwise_linear,
    reduce_packed_headwise_linear,
)
from ..joint_native.readers import TypedRead, _dense_reduce, _flat_messages, raw_scores
from .primitives import guarded_weights


def _empty_selected_values(reader, qstates, qcoords, coordinates, lengths,
                           scale, values, alpha, gate):
    """Keep explicit zero-gradient edges without evaluating omitted values."""
    inputs = (qstates, qcoords, coordinates, lengths, scale, values, alpha)
    zero = qstates.new_zeros(())
    for value in inputs:
        zero = zero + value.unsqueeze(0)[:0].sum()
    for name in ("query_value", "relative_value", "source_scale_projection", "message_output"):
        layer = getattr(reader, name)
        if layer is not None:
            for parameter in layer.parameters():
                zero = zero + parameter.unsqueeze(0)[:0].sum()
    batch, queries = qstates.shape[:2]
    context = zero.expand(batch, queries, reader.heads, reader.head_width)
    scalar_zero = zero + gate.unsqueeze(0)[:0].sum()
    if reader.thermal_output is not None:
        for parameter in reader.thermal_output.parameters():
            scalar_zero = scalar_zero + parameter.unsqueeze(0)[:0].sum()
    return context, scalar_zero.expand(0), zero.expand(0, reader.width)


def _flat_hidden(reader, qstates, qcoords, coordinates, lengths, scale, values, indices):
    """Evaluate only packed query/source pairs, before the output projection."""
    b, q, n = indices.unbind(-1)
    delta = qcoords[b, q] - coordinates[b, n]
    relative = reader.relative_value(_relative_gamma(delta, scale[b]))
    if reader.source_scale_projection is not None:
        relative = relative + reader.source_scale_projection(torch.asinh(delta / lengths[b, n]))
    return F.silu(reader.query_value(qstates)[b, q] + values[b, n] + relative)


def _relative_gamma(delta: Tensor, scale: Tensor) -> Tensor:
    relative = delta / scale
    pieces = [relative]
    for frequency in (1.0, 2.0, 4.0):
        phase = (2.0 * torch.pi * frequency) * relative
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(pieces, dim=-1)


def _dense_hidden(reader, qstates, qcoords, coordinates, lengths, scale, values):
    """Evaluate every rectangular query/source hidden value once."""
    descriptor, gamma = query_source_features(qcoords, coordinates, scale, lengths)
    relative = reader.relative_value(gamma)
    if reader.source_scale_projection is not None:
        relative = relative + reader.source_scale_projection(descriptor[..., -reader.coordinate_dim:])
    return F.silu(reader.query_value(qstates)[:, :, None] + values[:, None] + relative)


def _selected_source_order_reduce(reader, qstates, qcoords, coordinates, lengths, scale, values,
                                  indices, alpha, gate, return_messages=False):
    """Historical packed path, retained as a diagnostic backend."""
    batch, queries, count = alpha.shape[:3]
    if not indices.shape[0]:
        return _empty_selected_values(reader, qstates, qcoords, coordinates, lengths,
                                      scale, values, alpha, gate)
    messages, scalar = _flat_messages(reader, qstates, qcoords, coordinates, lengths, scale,
                                      values, indices)
    b, q, n = indices.unbind(-1)
    weighted = messages.reshape(-1, reader.heads, reader.head_width) * alpha[b, q, n, :, None]
    scratch = messages.new_zeros(batch, queries, count, reader.heads, reader.head_width)
    scratch.index_put_((b, q, n), weighted)
    context = scratch.sum(2)
    if reader.thermal_output is not None:
        scalar = scalar * gate[b, q, n]
    return context, scalar, messages if return_messages else messages.new_empty((0,))


def _segment_reduce(reader, qstates, qcoords, coordinates, lengths, scale, values,
                    indices, alpha, gate, return_messages=False):
    """Reduce packed hidden values by query/head before projecting messages."""
    batch, queries = qstates.shape[:2]
    if not indices.shape[0]:
        return _empty_selected_values(reader, qstates, qcoords, coordinates, lengths,
                                      scale, values, alpha, gate)

    hidden = _flat_hidden(reader, qstates, qcoords, coordinates, lengths, scale, values, indices)
    b, q, n = indices.unbind(-1)
    destination = b * queries + q
    selected_alpha = alpha[b, q, n]
    context = reduce_packed_headwise_linear(
        hidden, selected_alpha, destination, batch * queries, reader.message_output, reader.heads
    ).reshape(batch, queries, reader.heads, reader.head_width)

    if reader.thermal_output is None:
        scalar = hidden.new_empty((0,))
    else:
        weight, bias = compose_scalar_projection(reader.message_output, reader.thermal_output)
        scalar = F.linear(hidden, weight, bias).squeeze(-1) * gate[b, q, n]
    messages = reader.message_output(hidden) if return_messages else hidden.new_empty((0,))
    return context, scalar, messages


def _dense_weighted_reduce(reader, qstates, qcoords, coordinates, lengths, scale, values,
                           alpha, gate, return_messages=False):
    """Same-gate dense execution with hidden-first head reduction."""
    hidden = _dense_hidden(reader, qstates, qcoords, coordinates, lengths, scale, values)
    batch, queries, count = alpha.shape[:3]
    context = reduce_headwise_linear(hidden, alpha, reader.message_output, reader.heads)
    context = context.reshape(batch, queries, reader.heads, reader.head_width)
    if reader.thermal_output is None:
        scalar = hidden.new_empty((0,))
    else:
        weight, bias = compose_scalar_projection(reader.message_output, reader.thermal_output)
        scalar = F.linear(hidden, weight, bias).squeeze(-1) * gate
    messages = reader.message_output(hidden) if return_messages else hidden.new_empty((0,))
    return context, scalar, messages


def _counter(reference: Tensor, value: int) -> Tensor:
    """Create a compact device-side integer counter without a host reduction."""
    return torch.tensor(value, device=reference.device, dtype=torch.long)


def _work_counts(reader, qstates, valid, qmask, indices, backend, value_rows, selected_rows,
                 return_messages, checkpoint_values):
    batch, queries, width = qstates.shape
    count = valid.shape[1]
    candidate_rows = batch * queries * count
    present_rows = (valid.to(torch.long).sum(-1) * qmask.to(torch.long).sum(-1)).sum()
    legacy = backend in {"source_order", "dense"}
    if legacy:
        message_projection_rows = value_rows
        head_projection_rows = 0
        scratch_bytes = batch * queries * count * width * qstates.element_size() \
            if backend == "source_order" and indices.shape[0] else 0
    else:
        message_projection_rows = value_rows if return_messages else 0
        head_projection_rows = batch * queries * reader.heads
        scratch_bytes = 0
    if backend == "segment" and indices.shape[0] == 0:
        head_projection_rows = 0
    thermal_rows = value_rows if reader.thermal_output is not None else 0
    packed_bytes = 0
    if backend == "segment" and indices.shape[0]:
        selected_count = int(indices.shape[0])
        packed_bytes = (
            selected_count * width * qstates.element_size()
            + selected_count * reader.heads * qstates.element_size()
            + selected_count * 3 * torch.tensor([], dtype=torch.long).element_size()
        )
    checkpoint_rows = value_rows if checkpoint_values and torch.is_grad_enabled() else 0
    return dict(
        value_rows=_counter(qstates, value_rows),
        hidden_value_rows=_counter(qstates, value_rows),
        selected_rows=(selected_rows if isinstance(selected_rows, Tensor)
                       else _counter(qstates, int(selected_rows))),
        candidate_rows=_counter(qstates, candidate_rows),
        score_rows=_counter(qstates, candidate_rows),
        message_projection_rows=_counter(qstates, message_projection_rows),
        head_reduction_projection_rows=_counter(qstates, head_projection_rows),
        scalar_projection_rows=_counter(qstates, thermal_rows),
        fourier_entries=_counter(qstates, value_rows * 7 * reader.coordinate_dim),
        hidden_entries=_counter(qstates, value_rows * width),
        available_present_rows=present_rows,
        reduction_scratch_bytes=_counter(qstates, scratch_bytes),
        packed_intermediate_bytes=_counter(qstates, packed_bytes),
        padded_rows=_counter(qstates, 0),
        invalid_candidate_rows_available=_counter(qstates, 0),
        dense_value_rows=_counter(qstates, value_rows if backend in {"dense", "dense_weighted"} else 0),
        checkpoint_forward_rows=_counter(qstates, checkpoint_rows),
        checkpoint_recomputation_rows_available=_counter(qstates, 0),
    )


def read_continuous(reader, query, qstates, *, coordinates, lengths, valid, measure, frame_scale,
                    keys, values, gate, scores=None, backend="packed", return_messages=False,
                    frozen_coefficients=None, reference=None, checkpoint_values=False):
    """Read original records using a same-function packed or dense executor.

    ``segment`` reduces packed hidden values before their head-specific output
    projections. ``dense_weighted`` applies the same trained access and guarded
    weights to a regular dense hidden tensor. ``source_order`` and its legacy
    ``dense`` counterpart retain the historical projected-message reducers.
    ``packed`` remains an alias for ``source_order`` for old model configs.
    """
    if type(checkpoint_values) is not bool:
        raise ValueError("checkpoint_values must be a bool.")
    aliases = {"packed": "source_order", "weighted": "dense_weighted"}
    backend = aliases.get(backend, backend)
    if backend not in {"segment", "dense_weighted", "source_order", "dense"}:
        raise ValueError("Continuous backend must be segment, dense_weighted, source_order, or dense.")

    batch, queries, width = qstates.shape
    count = coordinates.shape[1]
    qmask = (query.query_mask if query.query_mask is not None else
             torch.ones((batch, queries), device=qstates.device, dtype=torch.bool))
    gate = gate * valid[:, None] * qmask[..., None]
    if frozen_coefficients is None and scores is None:
        scores = raw_scores(reader, query, qstates, coordinates, lengths, frame_scale, keys)
    if frozen_coefficients is None:
        alpha, _ = guarded_weights(scores, gate, measure, valid, qmask, reference=reference)
    else:
        alpha = frozen_coefficients
        if alpha.shape != (batch, queries, count, reader.heads):
            raise ValueError("Frozen read coefficients have incorrect native axes.")

    if backend in {"dense", "dense_weighted"}:
        indices = torch.empty((0, 3), device=qstates.device, dtype=torch.long)
        selected_rows = (gate.detach() > 0).sum(dtype=torch.long)
    else:
        indices = (gate.detach() > 0).nonzero(as_tuple=False)
        selected_rows = int(indices.shape[0])
    qcoords = query.coordinates.to(qstates)
    args = (reader, qstates, qcoords, coordinates.to(qstates), lengths.to(qstates), frame_scale, values)
    if backend == "source_order":
        function = lambda *inputs: _selected_source_order_reduce(
            *inputs, indices, alpha, gate, return_messages
        )
        if checkpoint_values and torch.is_grad_enabled() and indices.shape[0]:
            context, scalar, messages = checkpoint(
                function, *args, use_reentrant=False
            )
        else:
            context, scalar, messages = function(*args)
        value_rows = int(indices.shape[0])
        response = None
        if reader.thermal_output is not None:
            response = qstates.new_zeros(batch, queries, count)
            b, q, n = indices.unbind(-1)
            response = response.index_put((b, q, n), scalar)
        if return_messages:
            message_grid = qstates.new_zeros(batch, queries, count, width)
            b, q, n = indices.unbind(-1)
            messages = message_grid.index_put((b, q, n), messages)
    elif backend == "dense":
        function = lambda *inputs: _dense_reduce(
            *inputs, alpha, gate, return_messages
        )
        dense_args = (*args,)
        if checkpoint_values and torch.is_grad_enabled() and batch * queries * count:
            context, scalar, messages = checkpoint(
                function, *dense_args, use_reentrant=False
            )
        else:
            context, scalar, messages = function(*dense_args)
        value_rows = batch * queries * count
        response = scalar if reader.thermal_output is not None else None
    elif backend == "segment":
        function = lambda qs, qc, cs, ls, fs, vs, a, g: _segment_reduce(
            reader, qs, qc, cs, ls, fs, vs, indices, a, g, return_messages
        )
        segment_args = (*args[1:], alpha, gate)
        if checkpoint_values and torch.is_grad_enabled() and indices.shape[0]:
            context, scalar, messages = checkpoint(
                function, *segment_args, use_reentrant=False
            )
        else:
            context, scalar, messages = function(*segment_args)
        value_rows = int(indices.shape[0])
        response = None
        if reader.thermal_output is not None:
            response = qstates.new_zeros(batch, queries, count)
            b, q, n = indices.unbind(-1)
            response = response.index_put((b, q, n), scalar)
        if return_messages:
            message_grid = qstates.new_zeros(batch, queries, count, width)
            b, q, n = indices.unbind(-1)
            messages = message_grid.index_put((b, q, n), messages)
    else:
        function = lambda *inputs: _dense_weighted_reduce(
            *inputs, return_messages=return_messages
        )
        weighted_args = (*args, alpha, gate)
        if checkpoint_values and torch.is_grad_enabled() and batch * queries * count:
            context, scalar, messages = checkpoint(
                function, *weighted_args, use_reentrant=False
            )
        else:
            context, scalar, messages = function(*weighted_args)
        value_rows = batch * queries * count
        response = scalar if reader.thermal_output is not None else None

    counts = _work_counts(reader, qstates, valid, qmask, indices, backend, value_rows, selected_rows,
                          return_messages, checkpoint_values)
    return TypedRead(context.reshape(batch, queries, width), alpha, response,
                     messages if return_messages else None, counts)
