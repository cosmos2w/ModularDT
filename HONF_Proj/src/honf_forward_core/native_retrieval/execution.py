"""One optional tensor-only fusion of the unchanged selected value formula.

The pinned compiler is used for inference only. Autograd, including native
second derivatives, retains the eager reference. Compilation/setup costs are
separate from warm complete-call latency; no compilation cache is scientific
data or a token-value cache.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def selected_value_reduction(
    query_coordinates: Tensor,
    module_coordinates: Tensor,
    frame_scale: Tensor,
    query_projection: Tensor,
    module_projection: Tensor,
    score_descriptor: Tensor,
    pair_indices: Tensor,
    alpha: Tensor,
    scalar_permission: Tensor,
    relative_weight: Tensor,
    message_weight: Tensor,
    message_bias: Tensor,
    thermal_weight: Tensor | None,
    thermal_bias: Tensor | None,
    source_scale_weight: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gather original rows, interact nonlinearly, then reduce by receiver.

    Alpha has the maintained [B,Q,M,H] layout. No record-axis contraction
    occurs before the original query-dependent nonlinear message.
    """
    batch, queries, _modules, heads = alpha.shape
    width = query_projection.shape[-1]
    head_width = width // heads
    b, q, m = pair_indices.unbind(1)
    delta = query_coordinates[b, q] - module_coordinates[b, m]
    normalized = delta / frame_scale[b]
    pieces = [normalized]
    for frequency in (1., 2., 4.):
        phase = (2. * torch.pi * frequency) * normalized
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    relative = F.linear(torch.cat(pieces, -1), relative_weight)
    if source_scale_weight is not None:
        relative = relative + F.linear(score_descriptor[b, q, m, -delta.shape[-1]:], source_scale_weight)
    hidden = F.silu(query_projection[b, q] + module_projection[b, m] + relative)
    messages = F.linear(hidden, message_weight, message_bias)
    blocks = messages.reshape(-1, heads, head_width)
    weighted = blocks * alpha[b, q, m, :, None]
    context = messages.new_zeros((batch * queries, heads, head_width)).index_add(
        0, b * queries + q, weighted).reshape(batch, queries, width)
    if thermal_weight is None:
        coefficients = messages.new_empty((0,))
    else:
        coefficients = F.linear(messages, thermal_weight, thermal_bias).squeeze(-1) * scalar_permission
    return context, messages, coefficients


_compiled_selected_value_reduction = None


def compiled_selected_values(reader, organization, query, query_states, pair_indices, alpha, permission, scores):
    """Execute the fused inference kernel; caller retains the eager path.

    Compiler failures are explicit. They do not authorize a hidden numerical
    backend, smaller minibatch, or all-open replacement.
    """
    if torch.is_grad_enabled():
        raise ValueError("Compiled native values are inference-only; use the eager derivative reference.")
    global _compiled_selected_value_reduction
    if _compiled_selected_value_reduction is None:
        _compiled_selected_value_reduction = torch.compile(
            selected_value_reduction, fullgraph=True, dynamic=True)
    b, q, m = pair_indices.unbind(1)
    scalar_permission = permission.amax(1)[b, q, m]
    return _compiled_selected_value_reduction(
        query.coordinates.to(query_states.dtype), organization.scene.module_coordinates.to(query_states.dtype),
        organization.frame_scale, reader.query_value(query_states), organization.direct_source_values,
        scores.score_descriptor, pair_indices, alpha, scalar_permission,
        reader.relative_value.weight, reader.message_output.weight, reader.message_output.bias,
        None if reader.thermal_output is None else reader.thermal_output.weight,
        None if reader.thermal_output is None else reader.thermal_output.bias,
        None if reader.source_scale_projection is None else reader.source_scale_projection.weight)
