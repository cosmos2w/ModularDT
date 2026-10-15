"""Headwise readers and bounded local detail for content-hypergraph v2."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import QueryBatch
from .decoding import LocalPairPack, pack_local_pairs
from .source_read_execution import compose_scalar_projection, reduce_headwise_linear, reduce_indexed_linear
from .source_reader_variants import SourceScaleProjection
from .spatial_features_v2 import coordinate_gamma, query_edge_features, query_source_features
from .staged_attention_v2 import bounded_temperature, safe_masked_softmax


def _query_mask(query: QueryBatch) -> Tensor:
    if query.query_mask is None:
        return torch.ones(query.coordinates.shape[:2], dtype=torch.bool, device=query.coordinates.device)
    return query.query_mask


def _unit_gamma(values: Tensor) -> Tensor:
    pieces = [values]
    for frequency in (1.0, 2.0, 4.0):
        phase = 2.0 * torch.pi * frequency * values
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(pieces, dim=-1)


def _temperature_logit(head_width: int) -> float:
    fraction = (math.sqrt(float(head_width)) - 1.0) / 31.0
    if fraction <= 0.0:
        # The lower endpoint is only approached by a finite sigmoid logit.
        return -20.0
    return math.log(fraction / (1.0 - fraction))


class QueryEncoderV2(nn.Module):
    def __init__(self, query_input_dim: int, coordinate_dim: int, width: int) -> None:
        super().__init__()
        self.query_input_dim = int(query_input_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(query_input_dim + 7 * coordinate_dim + width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, query: QueryBatch, operating_state: Tensor, frame_origin: Tensor, frame_scale: Tensor):
        batch, query_count, _ = query.coordinates.shape
        if query.features is None:
            features = query.coordinates.new_zeros(batch, query_count, self.query_input_dim)
        else:
            features = query.features
        coordinates = query.coordinates.to(dtype=frame_scale.dtype)
        geometry = coordinate_gamma(coordinates, frame_origin, frame_scale)
        operation = operating_state[:, None, :].expand(batch, query_count, self.width)
        states = self.net(torch.cat((features.to(geometry.dtype), geometry, operation), dim=-1))
        return states * _query_mask(query)[:, :, None].to(states.dtype)


class LocalDetailReaderV2(nn.Module):
    """Same compact support, with multiscale relative values and width 128."""

    def __init__(
        self, width: int, coordinate_dim: int, hidden_width: int = 128, thermal: bool = True
    ) -> None:
        super().__init__()
        self.hidden_width = int(hidden_width)
        self.query_projection = nn.Linear(width, hidden_width, bias=False)
        self.module_projection = nn.Linear(width, hidden_width, bias=False)
        self.relative_projection = nn.Linear(7 * coordinate_dim, hidden_width, bias=False)
        self.flow_output = nn.Linear(hidden_width, hidden_width)
        self.thermal_output = nn.Linear(hidden_width, 1) if thermal else None

    def forward(
        self,
        scene,
        query: QueryBatch,
        module_states: Tensor,
        query_states: Tensor,
        frame_scale: Tensor,
        module_measure_reference: Tensor,
        local_gates: Tensor | None = None,
        execution_backend: str = "legacy",
    ) -> tuple[Tensor, LocalPairPack, Tensor]:
        if execution_backend not in {"legacy", "reduced_projection"}:
            raise ValueError("execution_backend must be 'legacy' or 'reduced_projection'.")
        batch, query_count, _ = query.coordinates.shape
        pack = pack_local_pairs(scene, query, frame_scale)
        pack.support_weight = pack.support_weight.to(device=query_states.device, dtype=query_states.dtype)
        local_context = query_states.new_zeros(batch * query_count, self.hidden_width)
        if pack.count == 0:
            return local_context.reshape(batch, query_count, self.hidden_width), pack, query_states.new_empty((0,))
        relative_geometry = _unit_gamma(pack.relative.to(query_states.dtype))
        if execution_backend == "reduced_projection":
            query_projection = self.query_projection(query_states)
            module_projection = self.module_projection(module_states)
            pair_pre_activation = (
                query_projection[pack.batch_index, pack.query_index]
                + module_projection[pack.batch_index, pack.module_index]
                + self.relative_projection(relative_geometry)
            )
        else:
            pair_pre_activation = (
                self.query_projection(query_states[pack.batch_index, pack.query_index])
                + self.module_projection(module_states[pack.batch_index, pack.module_index])
                + self.relative_projection(relative_geometry)
            )
        pair_hidden = F.silu(pair_pre_activation)
        module_measure = scene.module_measure[pack.batch_index, pack.module_index].to(query_states.dtype)
        measure_scale = module_measure_reference.to(device=query_states.device, dtype=query_states.dtype)
        gate = (
            torch.ones_like(pack.support_weight)
            if local_gates is None
            else local_gates[pack.batch_index, pack.module_index].to(query_states.dtype)
        )
        flow_weight = pack.support_weight * gate * module_measure / measure_scale[pack.batch_index].clamp_min(
            torch.finfo(query_states.dtype).tiny
        )
        flat_query = pack.batch_index * query_count + pack.query_index
        if execution_backend == "reduced_projection":
            local_context = reduce_indexed_linear(
                pair_hidden,
                flow_weight,
                flat_query,
                batch * query_count,
                self.flow_output,
            )
        else:
            flow_values = self.flow_output(pair_hidden)
            local_context = local_context.index_add(0, flat_query, flow_values * flow_weight[:, None])
        if self.thermal_output is None:
            thermal_coefficients = query_states.new_empty((0,))
        else:
            # Signed physical response coefficient; the smooth support is
            # applied once later during contraction with source controls.
            thermal_coefficients = self.thermal_output(pair_hidden).squeeze(-1) * gate
        return local_context.reshape(batch, query_count, self.hidden_width), pack, thermal_coefficients


class HeadwiseEdgeReaderV2(nn.Module):
    """One 128-wide query-edge message split into independently routed heads."""

    def __init__(
        self, width: int, coordinate_dim: int, thermal_rank: int, heads: int = 4,
        query_edge_geometry_mode: str | None = None,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("reader width must be divisible by feature-head count.")
        if query_edge_geometry_mode not in (None, "typed_w_centroids_v1"):
            raise ValueError("Unsupported query_edge_geometry_mode for the H6 reader.")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.thermal_rank = int(thermal_rank)
        self.query_edge_geometry_mode = query_edge_geometry_mode
        geometry_types = 1 if query_edge_geometry_mode is None else 3
        if query_edge_geometry_mode is not None:
            self.module_centroid_gate = nn.Parameter(torch.tensor(0.1))
            self.environment_centroid_gate = nn.Parameter(torch.tensor(0.1))
        self.query_norm = nn.LayerNorm(width)
        self.edge_norm = nn.LayerNorm(width)
        self.query_key = nn.Linear(width, width, bias=False)
        self.edge_key = nn.Linear(width, width, bias=False)
        self.raw_temperature = nn.Parameter(torch.full((heads,), _temperature_logit(self.head_width)))
        self.geometry_score = nn.Sequential(
            nn.Linear(geometry_types * (2 * coordinate_dim + 1), 32), nn.SiLU(), nn.Linear(32, heads, bias=False)
        )
        self.query_value = nn.Linear(width, width, bias=False)
        self.edge_value = nn.Linear(width, width, bias=False)
        self.relative_value = nn.Linear(geometry_types * 7 * coordinate_dim, width, bias=False)
        self.message_output = nn.Linear(width, width)
        self.thermal_heads = nn.ModuleList(
            nn.Linear(self.head_width, 1) for _ in range(heads)
        ) if thermal_rank else nn.ModuleList()

    @property
    def temperature(self) -> Tensor:
        return bounded_temperature(self.raw_temperature)

    def prepare_edges(self, edges: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.edge_norm(edges)
        return self.edge_key(normalized), self.edge_value(normalized)

    def _geometry_features(self, query: QueryBatch, query_states: Tensor, organization, frame_scale: Tensor):
        """Read typed final-W first moments without changing anchors or W.

        The optional channels describe the actual gathered donor coordinates,
        not physical ownership of the full residual/mixed content. All three
        descriptors share one narrow score MLP and one value projection.
        """
        coordinates = query.coordinates.to(query_states.dtype)
        if self.query_edge_geometry_mode is None:
            return query_edge_features(
                coordinates, organization.edge_anchors, frame_scale, organization.edge_scales,
            )
        names = ("module_content_centroids", "environment_content_centroids",
                 "module_content_valid", "environment_content_valid")
        if any(getattr(organization, name, None) is None for name in names):
            raise ValueError("Prepared typed content centroid geometry is missing.")
        centers = torch.cat((organization.edge_anchors, organization.module_content_centroids,
                             organization.environment_content_centroids), dim=1)
        geometry, gamma = query_edge_features(
            coordinates, centers, frame_scale, organization.edge_scales.repeat(1, 3, 1),
        )
        batch, query_count = coordinates.shape[:2]
        edge_count = organization.edges.shape[1]
        typed_scale = torch.stack((
            query_states.new_ones(batch, edge_count),
            self.module_centroid_gate * organization.module_content_valid.to(query_states.dtype),
            self.environment_centroid_gate * organization.environment_content_valid.to(query_states.dtype),
        ), dim=-1)[:, None, :, :, None]

        def concatenate_types(value: Tensor) -> Tensor:
            # The combined edge axis is [anchor H, module H, environment H].
            value = value.reshape(batch, query_count, 3, edge_count, value.shape[-1])
            value = value.permute(0, 1, 3, 2, 4) * typed_scale
            return value.flatten(-2)

        return concatenate_types(geometry), concatenate_types(gamma)

    def forward(
        self,
        query: QueryBatch,
        query_states: Tensor,
        organization,
        frame_scale: Tensor,
        *,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        return_messages: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        edges = organization.edges
        batch, query_count, _ = query_states.shape
        edge_count = edges.shape[1]
        if edge_count == 0 or query_count == 0:
            access = query_states.new_zeros(batch, query_count, edge_count, self.heads)
            messages = query_states.new_zeros(batch, query_count, edge_count, self.width) if return_messages else None
            basis = query_states.new_zeros(batch, query_count, edge_count, self.thermal_rank) if self.thermal_rank else None
            return query_states.new_zeros(batch, query_count, self.width), access, access, basis, messages
        if organization.edge_keys is None or organization.edge_values is None:
            raise ValueError("Prepared H6 edge projections are missing.")
        geometry, relative_gamma = self._geometry_features(query, query_states, organization, frame_scale)
        query_key = self.query_key(self.query_norm(query_states)).view(
            batch, query_count, self.heads, self.head_width
        )
        edge_key = organization.edge_keys.view(batch, edge_count, self.heads, self.head_width)
        query_key = F.normalize(query_key, dim=-1, eps=1.0e-6)
        edge_key = F.normalize(edge_key, dim=-1, eps=1.0e-6)
        logits = torch.einsum("bqrd,bhrd->bqhr", query_key, edge_key)
        logits = logits * self.temperature[None, None, None, :]
        logits = logits + self.geometry_score(geometry)
        query_allowed = _query_mask(query)[:, :, None, None]
        allowed = query_allowed.expand(batch, query_count, edge_count, self.heads)
        if uniform_access:
            learned = torch.full_like(logits, 1.0 / float(edge_count)) * allowed.to(logits.dtype)
        else:
            learned = safe_masked_softmax(logits.permute(0, 1, 3, 2), allowed.permute(0, 1, 3, 2), dim=-1)
            learned = learned.permute(0, 1, 3, 2)
        if edge_gates is None:
            gates = learned.new_ones(batch, edge_count)
        else:
            gates = edge_gates.to(dtype=learned.dtype)
        effective = learned * gates[:, None, :, None]

        query_value = self.query_value(query_states)
        edge_value = organization.edge_values
        relative_value = self.relative_value(relative_gamma)
        messages = self.message_output(
            F.silu(query_value[:, :, None, :] + edge_value[:, None, :, :] + relative_value)
        )
        feature_blocks = messages.view(batch, query_count, edge_count, self.heads, self.head_width)
        heads_context = torch.einsum("bqhr,bqhrd->bqrd", effective, feature_blocks)
        global_context = heads_context.reshape(batch, query_count, self.width)
        if self.thermal_rank:
            signed_modes = torch.cat(
                [self.thermal_heads[head](feature_blocks[:, :, :, head, :]) for head in range(self.heads)],
                dim=-1,
            )
            # One independent signed response mode per feature head.
            basis = effective * signed_modes
        else:
            basis = None
        return global_context, learned, effective, basis, messages if return_messages else None


class DirectSourceReaderV2(nn.Module):
    """A2-D information attention plus unnormalized signed source responses."""

    def __init__(
        self,
        width: int,
        coordinate_dim: int,
        heads: int = 4,
        thermal: bool = True,
        source_value_conditioning: str = "none",
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("reader width must be divisible by feature-head count.")
        self.width = int(width)
        self.coordinate_dim = int(coordinate_dim)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        if source_value_conditioning not in {"none", "source_scale"}:
            raise ValueError("source_value_conditioning must be 'none' or 'source_scale'.")
        self.source_value_conditioning = source_value_conditioning
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.query_key = nn.Linear(width, width, bias=False)
        self.source_key = nn.Linear(width, width, bias=False)
        self.raw_temperature = nn.Parameter(torch.full((heads,), _temperature_logit(self.head_width)))
        self.geometry_score = nn.Sequential(
            nn.Linear(2 * coordinate_dim + 1, 32), nn.SiLU(), nn.Linear(32, heads, bias=False)
        )
        self.query_value = nn.Linear(width, width, bias=False)
        self.source_value = nn.Linear(width, width, bias=False)
        self.relative_value = nn.Linear(7 * coordinate_dim, width, bias=False)
        self.source_scale_projection = (
            SourceScaleProjection(coordinate_dim, width)
            if source_value_conditioning == "source_scale"
            else None
        )
        self.message_output = nn.Linear(width, width)
        self.thermal_output = nn.Linear(width, 1) if thermal else None

    @property
    def temperature(self) -> Tensor:
        return bounded_temperature(self.raw_temperature)

    def prepare_sources(self, module_states: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.source_norm(module_states)
        return self.source_key(normalized), self.source_value(normalized)

    def forward(
        self,
        scene,
        query: QueryBatch,
        query_states: Tensor,
        frame_scale: Tensor,
        module_measure_reference: Tensor,
        prepared_keys: Tensor | None,
        prepared_values: Tensor | None,
        *,
        return_messages: bool = False,
        execution_backend: str = "legacy",
        message_query_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if execution_backend not in {"legacy", "reduced_projection"}:
            raise ValueError("execution_backend must be 'legacy' or 'reduced_projection'.")
        batch, query_count, _ = query_states.shape
        module_count = scene.num_modules
        if module_count == 0 or query_count == 0:
            access = query_states.new_zeros(batch, query_count, module_count, self.heads)
            direct = query_states.new_zeros(batch, query_count, module_count) if self.thermal_output is not None else None
            message_query_count = (
                query_count
                if execution_backend != "reduced_projection" or message_query_indices is None
                else int(message_query_indices.numel())
            )
            messages = (
                query_states.new_zeros(batch, message_query_count, module_count, self.width)
                if return_messages else None
            )
            return query_states.new_zeros(batch, query_count, self.width), access, direct, messages
        if prepared_keys is None or prepared_values is None:
            raise ValueError("Prepared A2-D source projections are missing.")
        geometry, relative_gamma = query_source_features(
            query.coordinates.to(query_states.dtype),
            scene.module_coordinates.to(query_states.dtype),
            frame_scale,
            scene.module_characteristic_lengths.to(query_states.dtype),
        )
        q = self.query_key(self.query_norm(query_states)).view(
            batch, query_count, self.heads, self.head_width
        )
        k = prepared_keys.view(batch, module_count, self.heads, self.head_width)
        q = F.normalize(q, dim=-1, eps=1.0e-6)
        k = F.normalize(k, dim=-1, eps=1.0e-6)
        logits = torch.einsum("bqrd,bmrd->bqmr", q, k)
        logits = logits * self.temperature[None, None, None, :]
        logits = logits + self.geometry_score(geometry)
        measure = scene.module_measure.to(query_states.dtype)
        valid = scene.module_mask & (measure > 0)
        safe_measure = torch.where(valid, measure, torch.ones_like(measure))
        reference = module_measure_reference.to(query_states.dtype).reshape(batch, 1)
        log_measure = torch.log(safe_measure) - torch.log(reference.clamp_min(torch.finfo(query_states.dtype).tiny))
        logits = logits + log_measure[:, None, :, None]
        allowed = valid[:, None, :, None].expand(batch, query_count, module_count, self.heads)
        allowed = allowed & _query_mask(query)[:, :, None, None]
        weights = safe_masked_softmax(logits.permute(0, 1, 3, 2), allowed.permute(0, 1, 3, 2), dim=-1)
        weights = weights.permute(0, 1, 3, 2)

        relative_value = self.relative_value(relative_gamma)
        if self.source_scale_projection is not None:
            # query_source_features already computed this asinh descriptor
            # for scoring; reuse its final coordinate_dim channels for values.
            source_scale = geometry[..., -self.coordinate_dim :]
            relative_value = relative_value + self.source_scale_projection(source_scale)
        hidden = F.silu(
            self.query_value(query_states)[:, :, None, :]
            + prepared_values[:, None, :, :]
            + relative_value
        )
        if execution_backend == "reduced_projection":
            context = reduce_headwise_linear(hidden, weights, self.message_output, self.heads)
            messages = None
        else:
            messages = self.message_output(hidden)
            blocks = messages.view(batch, query_count, module_count, self.heads, self.head_width)
            context = torch.einsum("bqmr,bqmrd->bqrd", weights, blocks).reshape(
                batch, query_count, self.width
            )
        # A2-D's thermal law is a physical signed source sum. This coefficient
        # is intentionally not multiplied by the information-attention weights.
        if self.thermal_output is None:
            direct_coefficients = None
        else:
            if execution_backend == "reduced_projection":
                composed_weight, composed_bias = compose_scalar_projection(
                    self.message_output, self.thermal_output
                )
                direct_coefficients = F.linear(hidden, composed_weight, composed_bias).squeeze(-1)
            else:
                direct_coefficients = self.thermal_output(messages).squeeze(-1)
            direct_coefficients = direct_coefficients * scene.module_mask[:, None, :].to(direct_coefficients.dtype)
            direct_coefficients = direct_coefficients * _query_mask(query)[:, :, None].to(direct_coefficients.dtype)
        if return_messages and messages is None:
            if message_query_indices is not None:
                selected_hidden = hidden.index_select(
                    1, message_query_indices.to(device=hidden.device, dtype=torch.long)
                )
            else:
                selected_hidden = hidden
            messages = self.message_output(selected_hidden)
        return context, weights, direct_coefficients, messages if return_messages else None


class FieldHeadV2(nn.Module):
    def __init__(self, width: int, hidden_width: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width * 2 + hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, output_dim),
        )

    def forward(self, query_state: Tensor, global_context: Tensor, local_context: Tensor) -> Tensor:
        return self.net(torch.cat((query_state, global_context, local_context), dim=-1))
