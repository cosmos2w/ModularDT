"""Two-pass, typed donor gathering and W-coupled thermal coefficients."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .organization import _reference_anchor_seeds
from .spatial_features_v2 import edge_pair_score_features, relative_gamma
from .staged_attention_v2 import bounded_temperature, safe_masked_softmax


def _temperature_logit() -> float:
    fraction = (math.sqrt(32.0) - 1.0) / 31.0
    return math.log(fraction / (1.0 - fraction))


def _measure_weighted_edge_read(
    edge_query: Tensor,
    donor_key: Tensor,
    donor_value: Tensor,
    geometry_bias: Tensor,
    anchors: Tensor,
    edge_scales: Tensor,
    donor_coordinates: Tensor,
    donor_lengths: Tensor,
    donor_measure: Tensor,
    measure_reference: Tensor,
    donor_mask: Tensor,
    frame_scale: Tensor,
    temperature: Tensor,
    *,
    geometry_prior: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Direct edge-to-donor W and typed content; empty types return exact zero."""
    batch, edge_count, width = edge_query.shape
    donor_count = donor_key.shape[1]
    if donor_count == 0:
        return (
            edge_query.new_zeros(batch, edge_count, 0),
            edge_query.new_zeros(batch, edge_count, width),
        )
    q = F.normalize(edge_query, dim=-1, eps=1.0e-6)
    k = F.normalize(donor_key, dim=-1, eps=1.0e-6)
    scores = torch.einsum("bhd,bnd->bhn", q, k) * temperature
    if geometry_prior is None:
        geometry, _relative, _ell_pair = edge_pair_score_features(
            anchors, donor_coordinates, frame_scale, edge_scales, donor_lengths
        )
        geometry_prior = geometry_bias(geometry).squeeze(-1)
        delta = anchors[:, :, None, :] - donor_coordinates[:, None, :, :]
        geometry_prior = geometry_prior - 0.5 * (delta / edge_scales[:, :, None, :]).square().sum(dim=-1)
    scores = scores + geometry_prior

    measure = donor_measure.to(dtype=scores.dtype)
    valid = donor_mask & (measure > 0)
    safe_measure = torch.where(valid, measure, torch.ones_like(measure))
    reference = measure_reference.to(dtype=scores.dtype).reshape(batch, 1)
    log_measure = torch.log(safe_measure) - torch.log(reference.clamp_min(torch.finfo(scores.dtype).tiny))
    scores = scores + log_measure[:, None, :]
    allowed = valid[:, None, :].expand(batch, edge_count, donor_count)

    # The normalized sum is small; FP64 accumulation keeps extreme positive
    # measures and score offsets stable while preserving live gradients.
    work_dtype = torch.float64 if scores.dtype == torch.float32 else scores.dtype
    weights = safe_masked_softmax(scores.to(work_dtype), allowed, dim=-1).to(scores.dtype)
    content = torch.einsum("bhn,bnd->bhd", weights, donor_value)
    return weights, content


class EdgeSelfAttentionV2(nn.Module):
    """One bounded H×H residual mixing step."""

    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("edge width must be divisible by attention heads.")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edges: Tensor, *, return_attention: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, count, _ = edges.shape
        if count == 0:
            attention = edges.new_zeros(batch, self.heads, 0, 0) if return_attention else None
            return edges, attention
        normalized = self.norm(edges)
        q = self.query(normalized).view(batch, count, self.heads, self.head_width).transpose(1, 2)
        k = self.key(normalized).view(batch, count, self.heads, self.head_width).transpose(1, 2)
        v = self.value(normalized).view(batch, count, self.heads, self.head_width).transpose(1, 2)
        updated = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )
        updated = updated.transpose(1, 2).reshape(batch, count, self.width)
        result = edges + self.dropout(self.output(updated))
        if not return_attention:
            return result, None
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width), dim=-1)
        return result, weights


class ContentBankV2(nn.Module):
    """Six (configurable H) spatially registered contents with two shared reads."""

    def __init__(
        self,
        width: int,
        coordinate_dim: int,
        edge_count: int = 6,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.coordinate_dim = int(coordinate_dim)
        self.edge_count = int(edge_count)
        self.heads = int(heads)
        self.edge_seeds = nn.Parameter(torch.empty(edge_count, width))
        self.anchor_offsets = nn.Parameter(torch.zeros(edge_count, coordinate_dim))
        self.raw_scale_fraction = nn.Parameter(
            torch.full((edge_count, coordinate_dim), math.log(0.2 / 0.8))
        )
        self.register_buffer("reference_anchor_seeds", _reference_anchor_seeds(edge_count, coordinate_dim))

        gamma_width = 7 * coordinate_dim
        self.initial_projection = nn.Sequential(
            nn.Linear(gamma_width + coordinate_dim + width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.initial_norm = nn.LayerNorm(width)

        # Both content reads call these same query/key/value/geometry modules.
        self.query_norm = nn.LayerNorm(width)
        self.edge_query = nn.Linear(width, width, bias=False)
        self.module_key_norm = nn.LayerNorm(width)
        self.environment_key_norm = nn.LayerNorm(width)
        self.module_key = nn.Linear(width, width, bias=False)
        self.environment_key = nn.Linear(width, width, bias=False)
        self.module_value = nn.Linear(width, width, bias=False)
        self.environment_value = nn.Linear(width, width, bias=False)
        score_width = 2 * coordinate_dim + 1
        self.module_geometry_score = nn.Sequential(
            nn.Linear(score_width, 32), nn.SiLU(), nn.Linear(32, 1, bias=False)
        )
        self.environment_geometry_score = nn.Sequential(
            nn.Linear(score_width, 32), nn.SiLU(), nn.Linear(32, 1, bias=False)
        )
        self.raw_gather_temperature = nn.Parameter(torch.tensor(_temperature_logit()))
        self.module_readout = nn.Linear(width, width, bias=False)
        self.environment_readout = nn.Linear(width, width, bias=False)
        self.first_norm = nn.LayerNorm(width)
        self.edge_interaction = EdgeSelfAttentionV2(width, heads, dropout)
        self.final_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    @property
    def gather_temperature(self) -> Tensor:
        return bounded_temperature(self.raw_gather_temperature)

    def _geometry(self, scene, frame_origin: Tensor, frame_scale: Tensor):
        reference = self.reference_anchor_seeds.to(device=frame_scale.device, dtype=frame_scale.dtype)
        normalized_anchors = (reference[None, :, :] + 0.05 * torch.tanh(self.anchor_offsets)[None, :, :]).expand(
            frame_scale.shape[0], -1, -1
        )
        anchors = frame_origin[:, None, :] + frame_scale[:, None, :] * normalized_anchors
        base = frame_scale[:, None, :] / (float(self.edge_count) ** (1.0 / self.coordinate_dim))
        fraction = 0.25 + 3.75 * torch.sigmoid(self.raw_scale_fraction)[None, :, :]
        scales = base * fraction
        return anchors, scales, normalized_anchors

    def _edge_query(self, edges: Tensor) -> Tensor:
        return self.edge_query(self.query_norm(edges))

    @staticmethod
    def _geometry_prior(
        anchors: Tensor,
        edge_scales: Tensor,
        donor_coordinates: Tensor,
        donor_lengths: Tensor,
        frame_scale: Tensor,
        geometry_score: nn.Module,
    ) -> Tensor:
        descriptor, _relative, _ell_pair = edge_pair_score_features(
            anchors, donor_coordinates, frame_scale, edge_scales, donor_lengths
        )
        geometry = geometry_score(descriptor).squeeze(-1)
        delta = anchors[:, :, None, :] - donor_coordinates[:, None, :, :]
        gaussian = -0.5 * (delta / edge_scales[:, :, None, :]).square().sum(dim=-1)
        return geometry + gaussian

    def forward(
        self,
        scene,
        module_states: Tensor,
        environment_states: Tensor,
        scene_summary: Tensor,
        operating_state: Tensor,
        frame_origin: Tensor,
        frame_scale: Tensor,
        module_measure_reference: Tensor,
        environment_measure_reference: Tensor,
        *,
        return_trace: bool = False,
    ) -> dict[str, Tensor]:
        batch = module_states.shape[0]
        dtype = module_states.dtype
        anchors, scales, normalized_anchors = self._geometry(scene, frame_origin, frame_scale)
        normalized_scale = scales / frame_scale[:, None, :]
        init_geometry = torch.cat(
            (
                self._gamma_from_normalized(normalized_anchors),
                torch.log(normalized_scale.clamp_min(torch.finfo(dtype).tiny)),
                scene_summary[:, None, :].expand(batch, self.edge_count, self.width),
            ),
            dim=-1,
        )
        initial = self.initial_norm(
            self.edge_seeds[None, :, :].to(dtype) + self.initial_projection(init_geometry)
        )

        module_key = self.module_key(self.module_key_norm(module_states))
        environment_key = self.environment_key(self.environment_key_norm(environment_states))
        module_value = self.module_value(module_states)
        environment_value = self.environment_value(environment_states)
        module_query = self._edge_query(initial)
        # Geometry and the weak anchor prior depend only on this scene, so
        # compute them once and reuse them for both shared-parameter reads.
        module_geometry_prior = self._geometry_prior(
            anchors,
            scales,
            scene.module_coordinates.to(dtype),
            scene.module_characteristic_lengths.to(dtype),
            frame_scale,
            self.module_geometry_score,
        )
        environment_geometry_prior = self._geometry_prior(
            anchors,
            scales,
            scene.environment_coordinates.to(dtype),
            scene.environment_characteristic_lengths.to(dtype),
            frame_scale,
            self.environment_geometry_score,
        )
        module_weights_first, module_content_first = _measure_weighted_edge_read(
            module_query,
            module_key,
            module_value,
            self.module_geometry_score,
            anchors,
            scales,
            scene.module_coordinates.to(dtype),
            scene.module_characteristic_lengths.to(dtype),
            scene.module_measure,
            module_measure_reference,
            scene.module_mask,
            frame_scale,
            self.gather_temperature,
            geometry_prior=module_geometry_prior,
        )
        environment_weights_first, environment_content_first = _measure_weighted_edge_read(
            module_query,
            environment_key,
            environment_value,
            self.environment_geometry_score,
            anchors,
            scales,
            scene.environment_coordinates.to(dtype),
            scene.environment_characteristic_lengths.to(dtype),
            scene.environment_measure,
            environment_measure_reference,
            scene.environment_mask,
            frame_scale,
            self.gather_temperature,
            geometry_prior=environment_geometry_prior,
        )
        first_content = self.first_norm(
            initial
            + self.dropout(self.module_readout(module_content_first))
            + self.dropout(self.environment_readout(environment_content_first))
        )

        mixed_content, edge_edge_attention = self.edge_interaction(first_content, return_attention=return_trace)
        second_query = self._edge_query(mixed_content)
        module_weights, module_content_second = _measure_weighted_edge_read(
            second_query,
            module_key,
            module_value,
            self.module_geometry_score,
            anchors,
            scales,
            scene.module_coordinates.to(dtype),
            scene.module_characteristic_lengths.to(dtype),
            scene.module_measure,
            module_measure_reference,
            scene.module_mask,
            frame_scale,
            self.gather_temperature,
            geometry_prior=module_geometry_prior,
        )
        environment_weights, environment_content_second = _measure_weighted_edge_read(
            second_query,
            environment_key,
            environment_value,
            self.environment_geometry_score,
            anchors,
            scales,
            scene.environment_coordinates.to(dtype),
            scene.environment_characteristic_lengths.to(dtype),
            scene.environment_measure,
            environment_measure_reference,
            scene.environment_mask,
            frame_scale,
            self.gather_temperature,
            geometry_prior=environment_geometry_prior,
        )
        # Explicit first-content residual preserves typed information around
        # the intervening H×H interaction and the second shared-parameter read.
        edges = self.final_norm(
            first_content
            + mixed_content
            + self.dropout(self.module_readout(module_content_second))
            + self.dropout(self.environment_readout(environment_content_second))
        )

        return {
            "edges": edges,
            "module_gather_weights": module_weights,
            "environment_gather_weights": environment_weights,
            "first_content": first_content,
            "mixed_content": mixed_content,
            "second_content": edges,
            "edge_anchors": anchors,
            "edge_scales": scales,
            "module_weights_first": module_weights_first,
            "environment_weights_first": environment_weights_first,
            "module_content_first": module_content_first,
            "environment_content_first": environment_content_first,
            "module_content_second": module_content_second,
            "environment_content_second": environment_content_second,
            "edge_edge_attention": edge_edge_attention
            if edge_edge_attention is not None
            else module_states.new_empty((0,)),
        }

    @staticmethod
    def _gamma_from_normalized(normalized: Tensor) -> Tensor:
        pieces = [normalized]
        for frequency in (1.0, 2.0, 4.0):
            phase = (2.0 * torch.pi * frequency) * normalized
            pieces.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(pieces, dim=-1)


class SourceCoefficientBankV2(nn.Module):
    """Signed source modes coupled exactly once to final module gather W."""

    def __init__(self, width: int, coordinate_dim: int, rank: int, hidden_width: int = 64) -> None:
        super().__init__()
        self.width = int(width)
        self.coordinate_dim = int(coordinate_dim)
        self.rank = int(rank)
        geometry_width = 7 * coordinate_dim
        self.edge_projection = nn.Linear(width, hidden_width, bias=False)
        self.module_projection = nn.Linear(width, hidden_width, bias=False)
        self.geometry_projection = nn.Linear(geometry_width, hidden_width, bias=False)
        self.scene_projection = nn.Linear(width, hidden_width, bias=False)
        self.operating_projection = nn.Linear(width, hidden_width, bias=False)
        self.output = nn.Linear(hidden_width, rank)

    def forward(
        self,
        scene,
        module_states: Tensor,
        edges: Tensor,
        module_gather_weights: Tensor,
        scene_summary: Tensor,
        operating_state: Tensor,
        edge_anchors: Tensor,
        frame_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, modules, _ = module_states.shape
        edge_count = edges.shape[1]
        if modules == 0 or edge_count == 0:
            kappa = module_states.new_zeros(batch, edge_count, modules, self.rank)
            coefficients = module_states.new_zeros(batch, edge_count, self.rank, modules)
            return kappa, coefficients
        delta = scene.module_coordinates.to(module_states.dtype)[:, None, :, :] - edge_anchors[:, :, None, :]
        geometry = relative_gamma(delta, frame_scale)
        hidden = self.edge_projection(edges)[:, :, None, :]
        hidden = hidden + self.module_projection(module_states)[:, None, :, :]
        hidden = hidden + self.geometry_projection(geometry)
        hidden = hidden + self.scene_projection(scene_summary)[:, None, None, :]
        hidden = hidden + self.operating_projection(operating_state)[:, None, None, :]
        kappa = self.output(F.silu(hidden))
        # W is the only global source gather factor in C; measure is already
        # present in W and is never multiplied into C a second time.
        coefficients = (module_gather_weights[:, :, :, None] * kappa).permute(0, 1, 3, 2)
        coefficients = coefficients * scene.module_mask[:, None, None, :].to(coefficients.dtype)
        return kappa, coefficients
