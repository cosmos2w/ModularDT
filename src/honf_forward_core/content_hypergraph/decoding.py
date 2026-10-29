"""Query-set consistent edge/source readers and bounded local detail."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import PreparedOrganization, QueryBatch, SceneBatch


@dataclass
class LocalPairPack:
    batch_index: Tensor
    query_index: Tensor
    module_index: Tensor
    support_weight: Tensor
    relative: Tensor

    @property
    def count(self) -> int:
        return int(self.batch_index.shape[0])


def _query_mask(query: QueryBatch) -> Tensor:
    if query.query_mask is None:
        return torch.ones(query.coordinates.shape[:2], dtype=torch.bool, device=query.coordinates.device)
    return query.query_mask


def pack_local_pairs(
    scene: SceneBatch,
    query: QueryBatch,
    frame_scale: Tensor,
) -> LocalPairPack:
    """Return only admitted local rows and their live smooth support weights."""
    _, n_query, coordinate_dim = query.coordinates.shape
    n_module = scene.module_coordinates.shape[1]
    qmask = _query_mask(query)
    if query.local_batch_indices is None:
        if n_query == 0 or n_module == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=query.coordinates.device)
            empty_weight = query.coordinates.new_empty((0,))
            empty_relative = query.coordinates.new_empty((0, coordinate_dim))
            return LocalPairPack(empty_long, empty_long, empty_long, empty_weight, empty_relative)
        query_coordinates = query.coordinates.to(dtype=frame_scale.dtype)
        module_coordinates = scene.module_coordinates.to(dtype=frame_scale.dtype)
        delta = query_coordinates[:, :, None, :] - module_coordinates[:, None, :, :]
        radius = scene.local_radius.to(dtype=frame_scale.dtype)[:, None, :].clamp_min(
            torch.finfo(frame_scale.dtype).tiny
        )
        ratio = torch.linalg.vector_norm(delta, dim=-1) / radius
        eligible = (ratio < 1.0) & scene.module_mask[:, None, :] & qmask[:, :, None]
        indices = torch.nonzero(eligible, as_tuple=False)
        if indices.numel() == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=query.coordinates.device)
            empty_weight = query.coordinates.new_empty((0,))
            empty_relative = query.coordinates.new_empty((0, coordinate_dim))
            return LocalPairPack(empty_long, empty_long, empty_long, empty_weight, empty_relative)
        b, q, m = indices.unbind(dim=1)
        selected_ratio = ratio[b, q, m]
        one_minus = (1.0 - selected_ratio).clamp_min(0.0)
        weights = one_minus.pow(4) * (4.0 * selected_ratio + 1.0)
        selected_delta = delta[b, q, m]
    else:
        b = query.local_batch_indices.to(device=query.coordinates.device, dtype=torch.long)
        q = query.local_query_indices.to(device=query.coordinates.device, dtype=torch.long)
        m = query.local_module_indices.to(device=query.coordinates.device, dtype=torch.long)
        in_bounds = qmask[b, q] & scene.module_mask[b, m]
        weights = query.local_weights * in_bounds.to(query.local_weights.dtype)
        selected_delta = query.coordinates[b, q].to(dtype=frame_scale.dtype) - scene.module_coordinates[b, m].to(
            dtype=frame_scale.dtype
        )

    radius = scene.local_radius[b, m].to(dtype=frame_scale.dtype).clamp_min(torch.finfo(frame_scale.dtype).tiny)
    relative = selected_delta / radius[:, None]
    # Supplied rows carry their own differentiable compact-window weights;
    # computed rows already use the same smooth endpoint-flat window.
    return LocalPairPack(b, q, m, weights, relative)


class QueryEncoder(nn.Module):
    def __init__(self, query_input_dim: int, coordinate_dim: int, width: int) -> None:
        super().__init__()
        self.query_input_dim = int(query_input_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(query_input_dim + coordinate_dim + width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(
        self,
        query: QueryBatch,
        operating_state: Tensor,
        frame_origin: Tensor,
        frame_scale: Tensor,
    ) -> Tensor:
        batch, query_count, _ = query.coordinates.shape
        if query.features is None:
            features = query.coordinates.new_zeros(batch, query_count, self.query_input_dim)
        else:
            features = query.features
        coordinates = query.coordinates.to(dtype=frame_scale.dtype)
        relative_coordinates = (coordinates - frame_origin[:, None, :]) / frame_scale[:, None, :]
        operation = operating_state[:, None, :].expand(batch, query_count, self.width)
        encoded = self.net(
            torch.cat((features.to(dtype=relative_coordinates.dtype), relative_coordinates, operation), dim=-1)
        )
        return encoded * _query_mask(query)[:, :, None].to(encoded.dtype)


class LocalDetailReader(nn.Module):
    def __init__(self, width: int, coordinate_dim: int, hidden_width: int = 64, thermal: bool = True) -> None:
        super().__init__()
        self.hidden_width = int(hidden_width)
        self.query_projection = nn.Linear(width, hidden_width, bias=False)
        self.module_projection = nn.Linear(width, hidden_width, bias=False)
        self.relative_projection = nn.Linear(coordinate_dim, hidden_width, bias=False)
        self.flow_output = nn.Linear(hidden_width, hidden_width)
        self.thermal_output = nn.Linear(hidden_width, 1) if thermal else None

    def forward(
        self,
        scene: SceneBatch,
        query: QueryBatch,
        module_states: Tensor,
        query_states: Tensor,
        frame_scale: Tensor,
        module_measure_reference: Tensor,
    ) -> tuple[Tensor, LocalPairPack, Tensor]:
        batch, query_count, _ = query.coordinates.shape
        pack = pack_local_pairs(scene, query, frame_scale)
        pack.support_weight = pack.support_weight.to(device=query_states.device, dtype=query_states.dtype)
        local_context = query_states.new_zeros(batch * query_count, self.hidden_width)
        if pack.count == 0:
            return local_context.reshape(batch, query_count, self.hidden_width), pack, query_states.new_empty((0,))
        operation = module_measure_reference.to(device=query_states.device, dtype=query_states.dtype)
        if operation.ndim == 1:
            operation = operation[:, None]
        module_measure = scene.module_measure[pack.batch_index, pack.module_index].to(query_states.dtype) / operation[
            pack.batch_index, 0
        ].clamp_min(torch.finfo(query_states.dtype).tiny)
        pair_hidden = F.silu(
            self.query_projection(query_states[pack.batch_index, pack.query_index])
            + self.module_projection(module_states[pack.batch_index, pack.module_index])
            + self.relative_projection(pack.relative)
        )
        flow_values = self.flow_output(pair_hidden)
        flat_query = pack.batch_index * query_count + pack.query_index
        flow_weight = pack.support_weight * module_measure
        local_context = local_context.index_add(0, flat_query, flow_values * flow_weight[:, None])
        if self.thermal_output is None:
            local_temperature_coefficients = query_states.new_empty((0,))
        else:
            local_temperature_coefficients = self.thermal_output(pair_hidden).squeeze(-1)
        return local_context.reshape(batch, query_count, self.hidden_width), pack, local_temperature_coefficients


class EdgeQueryReader(nn.Module):
    """Full-softmax H-edge query reader with optional affine spatial modes."""

    def __init__(self, width: int, coordinate_dim: int, hidden_width: int, thermal_rank: int) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden_width = int(hidden_width)
        self.thermal_rank = int(thermal_rank)
        self.query_norm = nn.LayerNorm(width)
        self.edge_norm = nn.LayerNorm(width)
        self.query_key = nn.Linear(width, width, bias=False)
        self.edge_key = nn.Linear(width, width, bias=False)
        self.geometry_score = nn.Linear(coordinate_dim + 1, 1, bias=False)
        self.query_value = nn.Linear(width, hidden_width, bias=False)
        self.edge_value = nn.Linear(width, hidden_width, bias=False)
        self.relative_value = nn.Linear(coordinate_dim, hidden_width, bias=False)
        self.message_output = nn.Linear(hidden_width, hidden_width)
        self.thermal_basis = nn.Linear(hidden_width, thermal_rank) if thermal_rank > 0 else None

    def prepare_edges(self, edges: Tensor) -> tuple[Tensor, Tensor]:
        """Project edge keys and content once during scene preparation."""
        normalized = self.edge_norm(edges)
        return self.edge_key(normalized), self.edge_value(normalized)

    def forward(
        self,
        query: QueryBatch,
        query_states: Tensor,
        organization: PreparedOrganization,
        *,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        edges = organization.edges
        batch, query_count, _ = query_states.shape
        edge_count = edges.shape[1]
        if edge_count == 0:
            empty_access = query_states.new_zeros(batch, query_count, 0)
            empty_basis = (
                query_states.new_zeros(batch, query_count, 0, self.thermal_rank) if self.thermal_rank else None
            )
            return (
                query_states.new_zeros(batch, query_count, self.hidden_width),
                empty_access,
                empty_access,
                empty_access,
                empty_basis,
                query_states.new_zeros(batch, query_count, 0, self.hidden_width),
            )
        query_coordinates = query.coordinates.to(dtype=query_states.dtype)
        relative = (
            query_coordinates[:, :, None, :] - organization.edge_anchors[:, None, :, :]
        ) / organization.edge_scales[:, None, :, :]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        geometry = torch.cat((relative, distance), dim=-1)
        q_key = self.query_key(self.query_norm(query_states))
        if organization.edge_keys is None or organization.edge_values is None:
            raise ValueError("Prepared edge projections are missing; rebuild the organization.")
        e_key = organization.edge_keys
        logits = torch.einsum("bqd,bhd->bqh", q_key, e_key) / math.sqrt(self.width)
        logits = logits + self.geometry_score(geometry).squeeze(-1)
        query_mask = _query_mask(query)[:, :, None].to(logits.dtype)
        learned_access = torch.softmax(logits, dim=-1) * query_mask
        if uniform_access:
            base_access = query_mask.expand(batch, query_count, edge_count) / float(edge_count)
        else:
            base_access = learned_access
        if edge_gates is None:
            gates = base_access.new_ones(batch, edge_count)
        else:
            if tuple(edge_gates.shape) != (batch, edge_count):
                raise ValueError(f"edge_gates must have shape {(batch, edge_count)}.")
            if edge_gates.device != base_access.device:
                raise ValueError("edge_gates must be on the prepared organization's device.")
            gates = edge_gates.to(dtype=base_access.dtype)
        effective_access = base_access * gates[:, None, :]
        messages = F.silu(
            self.query_value(query_states)[:, :, None, :]
            + organization.edge_values[:, None, :, :]
            + self.relative_value(relative)
        )
        messages = self.message_output(messages)
        global_context = torch.einsum("bqh,bqhw->bqw", effective_access, messages)
        if self.thermal_basis is None:
            basis = None
        else:
            basis = effective_access[..., None] * self.thermal_basis(messages)
        return global_context, learned_access, base_access, effective_access, basis, messages


class DirectSourceReader(nn.Module):
    """A-D source-resolved direct read, retained as the explicit control."""

    def __init__(self, width: int, coordinate_dim: int, hidden_width: int, thermal: bool) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden_width = int(hidden_width)
        self.query_norm = nn.LayerNorm(width)
        self.module_norm = nn.LayerNorm(width)
        self.query_key = nn.Linear(width, width, bias=False)
        self.module_key = nn.Linear(width, width, bias=False)
        self.geometry_score = nn.Linear(coordinate_dim + 1, 1, bias=False)
        self.query_value = nn.Linear(width, hidden_width, bias=False)
        self.module_value = nn.Linear(width, hidden_width, bias=False)
        self.relative_value = nn.Linear(coordinate_dim, hidden_width, bias=False)
        self.message_output = nn.Linear(hidden_width, hidden_width)
        self.thermal_output = nn.Linear(hidden_width, 1) if thermal else None

    def prepare_sources(self, module_states: Tensor) -> tuple[Tensor, Tensor]:
        """Project direct-source keys and values once during scene preparation."""
        normalized = self.module_norm(module_states)
        return self.module_key(normalized), self.module_value(normalized)

    def forward(
        self,
        scene: SceneBatch,
        query: QueryBatch,
        query_states: Tensor,
        module_states: Tensor,
        frame_scale: Tensor,
        module_measure_reference: Tensor,
        prepared_module_keys: Tensor | None = None,
        prepared_module_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        batch, query_count, _ = query_states.shape
        module_count = module_states.shape[1]
        if module_count == 0:
            access = query_states.new_zeros(batch, query_count, 0)
            messages = query_states.new_zeros(batch, query_count, 0, self.hidden_width)
            response = query_states.new_zeros(batch, query_count, 0) if self.thermal_output is not None else None
            return query_states.new_zeros(batch, query_count, self.hidden_width), access, response, messages
        query_coordinates = query.coordinates.to(dtype=query_states.dtype)
        module_coordinates = scene.module_coordinates.to(dtype=query_states.dtype)
        relative = (query_coordinates[:, :, None, :] - module_coordinates[:, None, :, :]) / frame_scale[
            :, None, None, :
        ]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        geometry = torch.cat((relative, distance), dim=-1)
        q_key = self.query_key(self.query_norm(query_states))
        if prepared_module_keys is None or prepared_module_values is None:
            raise ValueError("Prepared direct-source projections are required.")
        m_key = prepared_module_keys
        logits = torch.einsum("bqd,bmd->bqm", q_key, m_key) / math.sqrt(self.width)
        logits = logits + self.geometry_score(geometry).squeeze(-1)
        valid_modules = scene.module_mask & (scene.module_measure > 0)
        measure = scene.module_measure.to(dtype=query_states.dtype)
        safe_measure = torch.where(valid_modules, measure, torch.ones_like(measure))
        logits = logits + torch.log(safe_measure)[:, None, :]
        allowed = valid_modules[:, None, :] & _query_mask(query)[:, :, None]
        row_has_source = allowed.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~allowed, -torch.inf)
        logits = torch.where(row_has_source, logits, torch.zeros_like(logits))
        access = torch.softmax(logits, dim=-1) * allowed.to(logits.dtype)
        access = torch.where(row_has_source, access, torch.zeros_like(access))
        messages = F.silu(
            self.query_value(query_states)[:, :, None, :]
            + prepared_module_values[:, None, :, :]
            + self.relative_value(relative)
        )
        messages = self.message_output(messages)
        global_context = torch.einsum("bqm,bqmw->bqw", access, messages)
        if self.thermal_output is None:
            response = None
        else:
            response = access * self.thermal_output(messages).squeeze(-1)
        return global_context, access, response, messages


class FieldHead(nn.Module):
    def __init__(self, width: int, hidden_width: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_width + width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, output_dim),
        )

    def forward(self, query_state: Tensor, global_context: Tensor, local_context: Tensor) -> Tensor:
        return self.net(torch.cat((query_state, global_context, local_context), dim=-1))
