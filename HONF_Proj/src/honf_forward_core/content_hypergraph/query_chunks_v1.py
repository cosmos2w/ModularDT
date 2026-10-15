"""Explicit prepared-once receiver tiling for ordinary A2-D decoding.

Historical decoding has no internal query tile. Native observations are still
formed by the adapter after concatenating the complete decoded query union.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch

from .contracts import CoreOutput, QueryBatch


def slice_query_v1(query: QueryBatch, start: int, stop: int) -> QueryBatch:
    """Keep live tensor views and reindex supplied native local-pair rows."""
    if not 0 <= start <= stop <= query.num_queries:
        raise ValueError("Query slice is outside the original receiver axis.")
    values = {name: getattr(query, name)[:, start:stop] if getattr(query, name) is not None else None
              for name in ("coordinates", "features", "query_mask", "query_ids")}
    if query.local_query_indices is not None:
        keep = (query.local_query_indices >= start) & (query.local_query_indices < stop)
        for name in ("local_batch_indices", "local_module_indices", "local_weights"):
            values[name] = getattr(query, name)[keep]
        values["local_query_indices"] = query.local_query_indices[keep] - start
    # IDs and receiver ownership remain those of the original native request;
    # no geometry or support is rebuilt from a different query population.
    return replace(query, **values)


def decode_query_chunks_v1(
    query: QueryBatch, chunk_size: int, decode_validated: Callable[[QueryBatch], CoreOutput],
) -> CoreOutput:
    """Concatenate physical predictions; count actual work once per tile.

    Traces are intentionally handled by the core's explicit full-request trace
    path. This function never concatenates repeated preparation metadata or
    pretends that independently retained affine factors are one bound plan.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("reader_query_chunk_size must be a positive integer.")
    if query.num_queries <= chunk_size:
        return decode_validated(query)
    fields, temperatures, bases = [], [], []
    counts: dict[str, int] = {}
    for start in range(0, query.num_queries, chunk_size):
        output = decode_validated(slice_query_v1(query, start, min(start + chunk_size, query.num_queries)))
        if output.trace is not None:
            raise ValueError("Chunked decode does not accept a trace request.")
        fields.append(output.nonlinear_field)
        if output.thermal_temperature is not None:
            temperatures.append(output.thermal_temperature)
        if output.thermal_basis is not None:
            bases.append(output.thermal_basis)
        for name, value in (output.work_counts or {}).items():
            counts[name] = counts.get(name, 0) + int(value)
    counts["reader_query_tiles"] = len(fields)
    return CoreOutput(
        nonlinear_field=torch.cat(fields, dim=1),
        thermal_temperature=torch.cat(temperatures, dim=1) if temperatures else None,
        thermal_basis=torch.cat(bases, dim=1) if bases else None,
        work_counts=counts,
    )
