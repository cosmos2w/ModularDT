"""Differentiable contraction helpers for exact dense source reads."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def reduce_headwise_linear(
    hidden: Tensor,
    weights: Tensor,
    projection: nn.Linear,
    heads: int,
) -> Tensor:
    """Reduce hidden values by head before a shared linear output projection.

    ``weights`` may contain arbitrary nonnegative or signed coefficients. In
    particular, its mass is not assumed to be one: the output bias is scaled
    by the actual mass in each query/head row.
    """
    if hidden.ndim != 4 or weights.ndim != 4:
        raise ValueError("hidden and weights must have shapes [B,Q,M,D] and [B,Q,M,H].")
    batch, query_count, source_count, input_width = hidden.shape
    if tuple(weights.shape[:3]) != (batch, query_count, source_count):
        raise ValueError("weights must match hidden batch, query, and source axes.")
    if heads < 1 or weights.shape[-1] != heads:
        raise ValueError("weights head axis must match the positive heads argument.")
    if projection.in_features != input_width or projection.out_features % heads:
        raise ValueError("projection dimensions do not match hidden width and heads.")

    reduced = torch.einsum("bqmh,bqmd->bqhd", weights, hidden)
    head_width = projection.out_features // heads
    output_weight = projection.weight.reshape(heads, head_width, input_width)
    result = torch.einsum("bqhd,hcd->bqhc", reduced, output_weight)
    if projection.bias is not None:
        mass = weights.sum(dim=2)
        result = result + mass[..., None] * projection.bias.reshape(heads, head_width)
    return result.flatten(-2)


def compose_scalar_projection(message_output: nn.Linear, scalar_output: nn.Linear) -> tuple[Tensor, Tensor]:
    """Compose a scalar linear head with the message output projection.

    The returned tensors remain connected to both modules' parameters, so
    callers may use this during training and take optimizer steps normally.
    """
    if message_output.out_features != scalar_output.in_features or scalar_output.out_features != 1:
        raise ValueError("scalar_output must be a one-row projection over message_output outputs.")
    weight = scalar_output.weight @ message_output.weight
    bias = weight.new_zeros((1,))
    if scalar_output.bias is not None:
        bias = bias + scalar_output.bias
    if message_output.bias is not None:
        bias = bias + scalar_output.weight @ message_output.bias
    return weight, bias


def reduce_indexed_linear(
    hidden: Tensor,
    weights: Tensor,
    destination_index: Tensor,
    destination_count: int,
    projection: nn.Linear,
) -> Tensor:
    """Apply a linear projection after an indexed weighted reduction.

    This is algebraically equal to summing ``weights * projection(hidden)``
    by destination, including the mass-dependent projection bias.
    """
    if hidden.ndim != 2 or weights.ndim != 1 or destination_index.ndim != 1:
        raise ValueError("hidden, weights, and destination_index must be rank 2, 1, and 1.")
    if hidden.shape[0] != weights.numel() or weights.shape != destination_index.shape:
        raise ValueError("pair rows, weights, and destination indices must have equal lengths.")
    if hidden.shape[1] != projection.in_features or destination_count < 0:
        raise ValueError("projection input width and destination count are invalid.")
    if destination_index.numel():
        valid_index = ((destination_index >= 0) & (destination_index < destination_count)).all()
        if destination_index.is_cuda:
            # Keep the bounds assertion on-device; converting its result to a
            # Python bool would synchronize every reduced local read.
            torch._assert_async(valid_index)
        elif not bool(valid_index):
            raise ValueError("destination_index is outside the destination axis.")

    weighted = hidden * weights[:, None]
    reduced = hidden.new_zeros((destination_count, hidden.shape[-1])).index_add(
        0, destination_index, weighted
    )
    output = F.linear(reduced, projection.weight, bias=None)
    if projection.bias is not None:
        mass = weights.new_zeros((destination_count,)).index_add(0, destination_index, weights)
        output = output + mass[:, None] * projection.bias
    return output
