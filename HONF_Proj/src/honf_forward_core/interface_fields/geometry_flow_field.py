"""Small permutation-invariant, source-resolved geometry-to-query reader.

The caller owns physical feature semantics. This module never reads case IDs,
ports, thermal states, labels, or a reference generator.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn


def fourier_xy(x):
    frequency = x.new_tensor([math.pi * 2 ** i for i in range(4)])
    phase = x[..., None] * frequency
    return torch.cat((x, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


def mlp(input_width, output_width, hidden=128):
    return nn.Sequential(nn.Linear(input_width, hidden), nn.SiLU(), nn.Linear(hidden, output_width))


@dataclass
class PreparedGeometryFlow:
    context: torch.Tensor
    centers: torch.Tensor
    present: torch.Tensor
    lengths: torch.Tensor
    source_inputs: torch.Tensor
    source_states: torch.Tensor


class GeometryFlowField(nn.Module):
    """Two whole-layout source updates and dense nonlinear source reads."""
    def __init__(self, context_width=8, source_width=5, hidden=128, message=64, heat_columns=(4,)):
        super().__init__()
        self.config = {"context_width": context_width, "source_width": source_width,
                           "hidden": hidden, "message": message, "heat_columns": tuple(heat_columns)}
        self.source_encoder = mlp(source_width + context_width, hidden, hidden)
        # Explicit experimental columns start identically at zero in both arms.
        with torch.no_grad():
            self.source_encoder[0].weight[:, list(heat_columns)] = 0
        self.pair_messages = nn.ModuleList([mlp(2 * hidden + 18, message, hidden) for _ in range(2)])
        self.updates = nn.ModuleList([mlp(hidden + message + context_width, hidden, hidden) for _ in range(2)])
        self.source_read = nn.Linear(hidden, message)
        self.relative_read = mlp(18, message, message)
        self.read_output = mlp(message, message, message)
        self.background = mlp(18 + context_width, message, hidden)
        self.decoder = mlp(message + 18 + context_width, 4, hidden)

    def prepare(self, source_inputs, context, centers, present, lengths):
        mask = present[..., None]
        expanded_context = context[:, None].expand(-1, centers.shape[1], -1)
        state = self.source_encoder(torch.cat((source_inputs, expanded_context), -1)) * mask
        relative = fourier_xy((centers[:, :, None] - centers[:, None, :]) / lengths[:, None, None])
        count = present.sum(1, keepdim=True)
        pair_mask = present[:, :, None] * present[:, None, :]
        pair_mask = pair_mask * (1 - torch.eye(centers.shape[1], device=centers.device)[None])
        for message, update in zip(self.pair_messages, self.updates):
            left = state[:, :, None].expand(-1, -1, centers.shape[1], -1)
            right = state[:, None, :].expand(-1, centers.shape[1], -1, -1)
            messages = message(torch.cat((left, right, relative), -1)) * pair_mask[..., None]
            aggregate = messages.sum(2) / (count - 1).clamp_min(1)[..., None]
            state = (state + update(torch.cat((state, aggregate, expanded_context), -1))) * mask
        return PreparedGeometryFlow(context, centers, present, lengths, source_inputs, state)

    def read(self, prepared, query_xy, chunk_size=1024):
        outputs = []
        projected_sources = self.source_read(prepared.source_states)
        for query in query_xy.split(chunk_size, 1):
            query_features = fourier_xy(query / prepared.lengths[:, None])
            context = prepared.context[:, None].expand(-1, query.shape[1], -1)
            relative = fourier_xy((query[:, :, None] - prepared.centers[:, None]) / prepared.lengths[:, None, None])
            message = self.read_output(torch.nn.functional.silu(projected_sources[:, None] + self.relative_read(relative)))
            aggregate = (message * prepared.present[:, None, :, None]).sum(2)
            aggregate = aggregate / prepared.present.sum(1).clamp_min(1)[:, None, None]
            latent = aggregate + self.background(torch.cat((query_features, context), -1))
            outputs.append(self.decoder(torch.cat((latent, query_features, context), -1)))
        return torch.cat(outputs, 1)
