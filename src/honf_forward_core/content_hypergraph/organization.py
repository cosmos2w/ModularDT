"""Typed incidence, measure-weighted pooling, and source coefficients."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import SceneBatch


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / float(base)
    n = int(index)
    while n:
        value += float(n % base) * factor
        n //= base
        factor /= float(base)
    return value


def _reference_anchor_seeds(count: int, coordinate_dim: int) -> Tensor:
    primes = (2, 3, 5, 7, 11, 13, 17, 19)
    rows = [
        [_radical_inverse(edge + 1, primes[axis % len(primes)]) for axis in range(coordinate_dim)]
        for edge in range(count)
    ]
    # The coordinate frame is lower/span, so seeds must lie in its containing
    # unit box. Avoid boundaries where a compact-support response is exactly 0.
    return torch.tensor(rows, dtype=torch.float32).clamp(0.075, 0.925)


def _measure_weighted_pooling(
    scores: Tensor,
    measure: Tensor,
    mask: Tensor,
    measure_reference: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return incidence, normalized pooling, raw mass, log-mass feature and availability.

    ``scores`` has shape ``[B,N,H]``. The normalized pooling is evaluated in
    log space so a small positive edge mass does not create a reciprocal of
    that mass in the backward pass. The appended dummy donor is active only
    for a genuinely empty typed set; it makes the empty softmax finite while
    leaving every real donor weight exactly zero.
    """
    batch, nodes, edges = scores.shape
    dtype = scores.dtype
    measure = measure.to(dtype=dtype)
    valid = mask & (measure > 0)
    # Keep the learned projections/contents in their declared dtype. Evaluate
    # the small normalized sums in FP64 for FP32 inputs: subtracting a large
    # log assignment must not discard relative donor scores before softmax.
    accumulation_dtype = torch.float64 if dtype == torch.float32 else dtype
    safe_measure = torch.where(valid, measure, torch.ones_like(measure)).to(accumulation_dtype)
    log_incidence = F.log_softmax(scores.to(accumulation_dtype), dim=-1)
    log_weight_mass = log_incidence + torch.log(safe_measure)[:, :, None]
    log_weight_mass = log_weight_mass.masked_fill(~valid[:, :, None], -torch.inf).transpose(1, 2)

    has_donor = valid.any(dim=-1)
    dummy_log_weight = torch.where(
        has_donor[:, None, None],
        torch.full((batch, edges, 1), -torch.inf, dtype=accumulation_dtype, device=scores.device),
        torch.zeros((batch, edges, 1), dtype=accumulation_dtype, device=scores.device),
    )
    safe_log_weight_mass = torch.cat((log_weight_mass, dummy_log_weight), dim=-1)
    pooling = torch.softmax(safe_log_weight_mass, dim=-1)[..., :nodes]
    log_edge_mass = torch.logsumexp(safe_log_weight_mass, dim=-1)

    # softplus(log(D) - log(reference)) is log1p(D / reference), without
    # materializing a tiny D/reference ratio or adding an artificial mass.
    reference = measure_reference.to(device=scores.device, dtype=accumulation_dtype).reshape(batch, 1)
    reference = reference.clamp_min(torch.finfo(dtype).tiny)
    log_mass_feature = F.softplus(log_edge_mass - torch.log(reference))
    log_mass_feature = torch.where(has_donor[:, None], log_mass_feature, torch.zeros_like(log_mass_feature))

    incidence = torch.softmax(scores, dim=-1) * mask[:, :, None].to(dtype)
    physical_measure = torch.where(valid, measure, torch.zeros_like(measure))
    weight_mass = incidence.transpose(1, 2) * physical_measure[:, None, :]
    edge_mass = weight_mass.sum(dim=-1)
    stable_pooling = pooling.to(dtype)
    stable_log_mass = log_mass_feature.to(dtype)
    # Preserve the already measured FP32 forward rounding on normal masses.
    # The two expressions are the same analytic normalized sum: only their
    # floating-point rounding differs. Differentiate its stable expression,
    # avoiding the tiny reciprocal in the legacy quotient's backward pass.
    # For zero/subnormal masses use the stable value as well. This threshold
    # is the dtype's normal range, not a physical mass/occupancy regularizer.
    normal_mass = edge_mass >= torch.finfo(dtype).tiny
    legacy_denom = torch.where(normal_mass, edge_mass, torch.ones_like(edge_mass))
    legacy_pooling = weight_mass / legacy_denom[:, :, None]
    legacy_reference = measure_reference.to(device=scores.device, dtype=dtype).reshape(batch, 1)
    legacy_reference = legacy_reference.clamp_min(torch.finfo(dtype).tiny)
    legacy_log_mass = torch.log1p(edge_mass / legacy_reference)
    forward_pooling = torch.where(normal_mass[:, :, None], legacy_pooling, stable_pooling)
    forward_log_mass = torch.where(normal_mass, legacy_log_mass, stable_log_mass)
    pooling = forward_pooling.detach() + (stable_pooling - stable_pooling.detach())
    log_mass_feature = forward_log_mass.detach() + (stable_log_mass - stable_log_mass.detach())
    available = has_donor[:, None].expand(batch, edges)
    return incidence, pooling, edge_mass, log_mass_feature, available


class HypergraphOrganizer(nn.Module):
    """Prepare content-bearing edge states from contextualized native tokens."""

    def __init__(self, width: int, edge_count: int, coordinate_dim: int) -> None:
        super().__init__()
        self.width = int(width)
        self.edge_count = int(edge_count)
        self.coordinate_dim = int(coordinate_dim)
        self.edge_seeds = nn.Parameter(torch.empty(edge_count, width))
        self.anchor_offsets = nn.Parameter(torch.zeros(edge_count, coordinate_dim))
        self.log_edge_scales = nn.Parameter(torch.zeros(edge_count, coordinate_dim))
        self.register_buffer("reference_anchor_seeds", _reference_anchor_seeds(edge_count, coordinate_dim))

        self.module_key = nn.Linear(width, width, bias=False)
        self.environment_key = nn.Linear(width, width, bias=False)
        self.edge_query = nn.Linear(width, width, bias=False)
        self.module_geometry_bias = nn.Linear(coordinate_dim + 1, 1, bias=False)
        self.environment_geometry_bias = nn.Linear(coordinate_dim + 1, 1, bias=False)
        self.module_content = nn.Linear(width, width, bias=False)
        self.environment_content = nn.Linear(width, width, bias=False)
        edge_input_width = 3 * width + coordinate_dim + 4
        self.edge_fusion = nn.Sequential(
            nn.Linear(edge_input_width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.edge_norm = nn.LayerNorm(width)

    def forward(
        self,
        scene: SceneBatch,
        module_states: Tensor,
        environment_states: Tensor,
        operating_state: Tensor,
        frame_origin: Tensor,
        frame_scale: Tensor,
    ) -> dict[str, Tensor]:
        batch = module_states.shape[0]
        dtype = module_states.dtype
        device = module_states.device
        reference = self.reference_anchor_seeds.to(device=device, dtype=dtype)
        offsets = 0.05 * torch.tanh(self.anchor_offsets).to(dtype=dtype)
        anchors = frame_origin[:, None, :] + frame_scale[:, None, :] * (reference[None, :, :] + offsets[None, :, :])
        edge_scales = frame_scale[:, None, :] * torch.exp(self.log_edge_scales.to(dtype=dtype))[None, :, :]
        edge_scales = edge_scales.clamp_min(torch.finfo(dtype).tiny)

        edge_query = self.edge_query(self.edge_norm(self.edge_seeds)).to(dtype=dtype)
        module_key = self.module_key(module_states)
        environment_key = self.environment_key(environment_states)
        module_coordinates = scene.module_coordinates.to(dtype=dtype)
        environment_coordinates = scene.environment_coordinates.to(dtype=dtype)
        module_rel = (module_coordinates[:, :, None, :] - anchors[:, None, :, :]) / frame_scale[:, None, None, :]
        environment_rel = (environment_coordinates[:, :, None, :] - anchors[:, None, :, :]) / frame_scale[
            :, None, None, :
        ]
        module_geom = torch.cat((module_rel, torch.linalg.vector_norm(module_rel, dim=-1, keepdim=True)), dim=-1)
        environment_geom = torch.cat(
            (environment_rel, torch.linalg.vector_norm(environment_rel, dim=-1, keepdim=True)), dim=-1
        )
        module_scores = torch.einsum("bmd,hd->bmh", module_key, edge_query) / math.sqrt(self.width)
        environment_scores = torch.einsum("bed,hd->beh", environment_key, edge_query) / math.sqrt(self.width)
        module_scores = module_scores + self.module_geometry_bias(module_geom).squeeze(-1)
        environment_scores = environment_scores + self.environment_geometry_bias(environment_geom).squeeze(-1)

        if scene.module_measure_reference is None:
            module_reference = module_states.new_ones(batch)
        else:
            module_reference = scene.module_measure_reference.to(dtype).reshape(batch)
        if scene.environment_measure_reference is None:
            environment_reference = environment_states.new_ones(batch)
        else:
            environment_reference = scene.environment_measure_reference.to(dtype).reshape(batch)

        (
            module_incidence,
            module_pooling,
            module_edge_mass,
            module_log_mass,
            module_available,
        ) = _measure_weighted_pooling(
            module_scores, scene.module_measure, scene.module_mask, module_reference
        )
        (
            environment_incidence,
            environment_pooling,
            environment_edge_mass,
            environment_log_mass,
            environment_available,
        ) = _measure_weighted_pooling(
            environment_scores, scene.environment_measure, scene.environment_mask, environment_reference
        )

        module_summary = torch.einsum("bhm,bmd->bhd", module_pooling, self.module_content(module_states))
        environment_summary = torch.einsum(
            "bhe,bed->bhd", environment_pooling, self.environment_content(environment_states)
        )
        type_available = torch.stack((module_available, environment_available), dim=-1).to(dtype)
        anchors_normalized = (anchors - frame_origin[:, None, :]) / frame_scale[:, None, :]
        op = operating_state[:, None, :].expand(batch, self.edge_count, self.width)
        fusion_input = torch.cat(
            (
                module_summary,
                environment_summary,
                op,
                anchors_normalized,
                module_log_mass[:, :, None],
                environment_log_mass[:, :, None],
                type_available,
            ),
            dim=-1,
        )
        edges = self.edge_seeds[None, :, :].to(dtype) + self.edge_fusion(fusion_input)
        edges = self.edge_norm(edges)
        return {
            "edges": edges,
            "module_incidence": module_incidence,
            "environment_incidence": environment_incidence,
            "module_pooling_weights": module_pooling,
            "environment_pooling_weights": environment_pooling,
            "module_edge_mass": module_edge_mass,
            "environment_edge_mass": environment_edge_mass,
            "edge_anchors": anchors,
            "edge_scales": edge_scales,
        }


class SourceCoefficientBank(nn.Module):
    """Prepare signed source-indexed affine response factors ``C[B,H,R,M]``."""

    def __init__(self, width: int, coordinate_dim: int, rank: int, hidden_width: int = 64) -> None:
        super().__init__()
        self.width = int(width)
        self.coordinate_dim = int(coordinate_dim)
        self.rank = int(rank)
        self.edge_projection = nn.Linear(width, hidden_width, bias=False)
        self.module_projection = nn.Linear(width, hidden_width, bias=False)
        self.geometry_projection = nn.Linear(coordinate_dim + 1, hidden_width, bias=False)
        self.operating_projection = nn.Linear(width, hidden_width, bias=False)
        self.output = nn.Linear(hidden_width, rank)

    def forward(
        self,
        scene: SceneBatch,
        module_states: Tensor,
        edges: Tensor,
        module_incidence: Tensor,
        operating_state: Tensor,
        edge_anchors: Tensor,
        frame_scale: Tensor,
    ) -> Tensor:
        batch, modules, _ = module_states.shape
        edge_count = edges.shape[1]
        if modules == 0 or edge_count == 0:
            return module_states.new_zeros(batch, edge_count, self.rank, modules)
        module_coordinates = scene.module_coordinates.to(dtype=module_states.dtype)
        relative = (module_coordinates[:, None, :, :] - edge_anchors[:, :, None, :]) / frame_scale[:, None, None, :]
        geometry = torch.cat((relative, torch.linalg.vector_norm(relative, dim=-1, keepdim=True)), dim=-1)
        hidden = self.edge_projection(edges)[:, :, None, :]
        hidden = hidden + self.module_projection(module_states)[:, None, :, :]
        hidden = hidden + self.geometry_projection(geometry)
        hidden = hidden + self.operating_projection(operating_state)[:, None, None, :]
        coefficients = self.output(F.silu(hidden)).permute(0, 1, 3, 2)
        coefficients = coefficients * module_incidence.transpose(1, 2)[:, :, None, :]
        coefficients = coefficients * scene.module_mask[:, None, None, :].to(coefficients.dtype)
        return coefficients
