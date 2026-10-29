"""Selected fine-value evaluation and exact selected-function reductions.

The all-source inputs contain only small geometry/score tensors. Fine Fourier
features and 128-wide hidden values are built after the physical-pair union
is selected, except for the explicit dense-masked arithmetic reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .source_read_plans import SourceReadPlanV1, build_degree_bucket_packet_layout
from .decoding import _query_mask
from .staged_attention_v2 import safe_masked_softmax


@dataclass(frozen=True)
class FineSourceRowsV1:
    hidden: Tensor
    counts: dict[str, int]


def source_score_descriptor_v1(query: Tensor, source: Tensor, scale: Tensor, lengths: Tensor) -> Tensor:
    """Existing A2-D small descriptor, without constructing fine Fourier rows."""
    delta = query[:, :, None, :] - source[:, None, :, :]
    normalized = delta / scale[:, None, None, :]
    distance = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
    return torch.cat((normalized, distance, torch.asinh(delta / lengths[:, None, :, :])), dim=-1)


def original_source_scores_v1(reader, scene, query, query_states, frame_scale, measure_reference, prepared_keys):
    """Compute the original all-source attention before fine-value geometry."""
    batch, count = query_states.shape[:2]
    sources = scene.num_modules
    descriptor = source_score_descriptor_v1(
        query.coordinates.to(query_states.dtype), scene.module_coordinates.to(query_states.dtype),
        frame_scale, scene.module_characteristic_lengths.to(query_states.dtype))
    if not sources or not count:
        return descriptor, query_states.new_zeros((batch, count, sources, reader.heads))
    q = F.normalize(reader.query_key(reader.query_norm(query_states)).reshape(
        batch, count, reader.heads, reader.head_width), dim=-1, eps=1e-6)
    k = F.normalize(prepared_keys.reshape(batch, sources, reader.heads, reader.head_width), dim=-1, eps=1e-6)
    logits = torch.einsum("bqrd,bmrd->bqmr", q, k) * reader.temperature[None, None, None, :]
    logits = logits + reader.geometry_score(descriptor)
    measure = scene.module_measure.to(query_states.dtype)
    valid = scene.module_mask & (measure > 0)
    safe_measure = torch.where(valid, measure, torch.ones_like(measure))
    reference = measure_reference.to(query_states.dtype).reshape(batch, 1)
    logits = logits + (torch.log(safe_measure) - torch.log(reference.clamp_min(
        torch.finfo(query_states.dtype).tiny)))[:, None, :, None]
    allowed = valid[:, None, :, None].expand(batch, count, sources, reader.heads)
    allowed = allowed & _query_mask(query)[:, :, None, None]
    alpha = safe_masked_softmax(logits.permute(0, 1, 3, 2), allowed.permute(0, 1, 3, 2), dim=-1)
    return descriptor, alpha.permute(0, 1, 3, 2)


def _fine_hidden(reader, descriptor: Tensor, query_value: Tensor, source_value: Tensor) -> Tensor:
    normalized = descriptor[..., :reader.coordinate_dim]
    pieces = [normalized]
    for frequency in (1.0, 2.0, 4.0):
        phase = (2.0 * torch.pi * frequency) * normalized
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    relative = reader.relative_value(torch.cat(pieces, dim=-1))
    if reader.source_scale_projection is not None:
        relative = relative + reader.source_scale_projection(descriptor[..., -reader.coordinate_dim:])
    return F.silu(query_value + source_value + relative)


def evaluate_fine_source_rows_v1(
    reader, plan: SourceReadPlanV1, descriptor: Tensor, query_values: Tensor, source_values: Tensor,
    *, executor: str,
) -> FineSourceRowsV1:
    """Return fine hidden rows in canonical packed order, including packet cost.

    A degree bucket keeps each receiver's original source list and evaluates
    padded slots as real kernel work. It never substitutes a shared superset.
    """
    pairs = plan.packed_pairs
    b, q, m = pairs.batch_index, pairs.query_index, pairs.source_index
    selected = pairs.pair_count
    counts = {"fine_selected_rows": selected, "fine_executed_rows": selected,
              "fine_padding_rows": 0, "packet_sort_rows": 0,
              "fine_fourier_feature_entries": selected * 7 * reader.coordinate_dim,
              "fine_hidden_entries": selected * reader.width}
    if executor == "packed":
        hidden = _fine_hidden(reader, descriptor[b, q, m], query_values[b, q], source_values[b, m])
    elif executor == "dense_masked":
        # Deliberate same-function reference, never reported as skipped work.
        all_hidden = _fine_hidden(reader, descriptor, query_values[:, :, None], source_values[:, None])
        hidden = all_hidden[b, q, m]
        executed = plan.batch_size * plan.query_count * plan.source_count
        counts.update(fine_executed_rows=executed, fine_hidden_entries=executed * reader.width,
                      fine_fourier_feature_entries=executed * 7 * reader.coordinate_dim)
    elif executor == "packet":
        layout = build_degree_bucket_packet_layout(plan)
        values, identities = [], []
        for bucket in layout.buckets:
            pb = bucket.batch_index[:, None].expand_as(bucket.source_indices)
            pq = bucket.query_index[:, None].expand_as(bucket.source_indices)
            pm = bucket.source_indices.clamp_min(0)
            admitted = ~bucket.padding_mask
            fine = _fine_hidden(reader, descriptor[pb, pq, pm], query_values[pb, pq], source_values[pb, pm])
            values.append(fine[admitted])
            identities.append(((pb * plan.query_count + pq) * plan.source_count + pm)[admitted])
        if values:
            concatenated = torch.cat(values, dim=0)
            order = torch.argsort(torch.cat(identities), stable=True)
            hidden = concatenated.index_select(0, order)
        else:
            hidden = query_values.new_empty((0, reader.width))
        executed = layout.allocated_pair_slots
        counts.update(fine_executed_rows=executed, fine_padding_rows=layout.padding_pair_count,
                      packet_sort_rows=selected, packet_bucket_count=len(layout.buckets),
                      packet_query_assignment_rows=layout.bucket_assignment_query_count,
                      fine_hidden_entries=executed * reader.width,
                      fine_fourier_feature_entries=executed * 7 * reader.coordinate_dim)
    else:
        raise ValueError("source_read_executor must be packed, packet, or dense_masked.")
    return FineSourceRowsV1(hidden=hidden, counts=counts)


def reduce_selected_headwise_v1(
    hidden: Tensor, plan: SourceReadPlanV1, projection: nn.Linear,
    *, base_head_hidden: Tensor | None = None,
) -> Tensor:
    """Project after selected hidden reduction, with correct original mass bias.

    For completion, ``hidden`` contains selected fine-minus-base rows and
    ``base_head_hidden`` is the cheap all-source headwise contraction. Its
    output bias therefore uses full original alpha mass exactly once.
    """
    pairs = plan.packed_pairs
    heads = plan.original_alpha.shape[-1]
    batch, queries = plan.batch_size, plan.query_count
    if tuple(hidden.shape) != (pairs.pair_count, projection.in_features):
        raise ValueError("Selected hidden shape differs from the bound source plan.")
    if projection.out_features % heads:
        raise ValueError("Output projection width must be divisible by the head count.")
    flat_query = pairs.batch_index * queries + pairs.query_index
    weights = pairs.original_alpha * pairs.gate[:, None]
    rows, masses = [], []
    for head in range(heads):
        weighted_hidden = hidden * weights[:, head, None]
        rows.append(hidden.new_zeros((batch * queries, hidden.shape[-1])).index_add(0, flat_query, weighted_hidden))
        masses.append(weights.new_zeros((batch * queries,)).index_add(0, flat_query, weights[:, head]))
    reduced = torch.stack(rows, dim=1).reshape(batch, queries, heads, hidden.shape[-1])
    mass = torch.stack(masses, dim=1).reshape(batch, queries, heads)
    if base_head_hidden is not None:
        if base_head_hidden.shape != reduced.shape:
            raise ValueError("Completion headwise contraction has the wrong shape.")
        reduced = reduced + base_head_hidden
        mass = plan.original_alpha.sum(dim=2)
    output_width = projection.out_features // heads
    result = torch.einsum("bqhd,hcd->bqhc", reduced, projection.weight.reshape(heads, output_width, -1))
    if projection.bias is not None:
        result = result + mass[..., None] * projection.bias.reshape(heads, output_width)
    return result.flatten(-2)
