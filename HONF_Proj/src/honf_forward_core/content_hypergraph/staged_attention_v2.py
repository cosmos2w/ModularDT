"""Ordered, geometry-aware token exchange for content-hypergraph v2."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spatial_features_v2 import pair_score_features


def bounded_temperature(raw: Tensor) -> Tensor:
    """Map live unconstrained parameters to the declared closed interval."""
    return 1.0 + 31.0 * torch.sigmoid(raw)


def _initial_temperature_logit(head_width: int = 32) -> float:
    fraction = (math.sqrt(float(head_width)) - 1.0) / 31.0
    return math.log(fraction / (1.0 - fraction))


def safe_masked_softmax(logits: Tensor, allowed: Tensor, dim: int = -1) -> Tensor:
    """Softmax with exact zero for empty rows and no all-masked NaNs."""
    if logits.shape[dim] == 0:
        return torch.zeros_like(logits)
    allowed = allowed.to(torch.bool)
    row_has_donor = allowed.any(dim=dim, keepdim=True)
    masked = logits.masked_fill(~allowed, -torch.inf)
    masked = torch.where(row_has_donor, masked, torch.zeros_like(masked))
    weights = torch.softmax(masked, dim=dim) * allowed.to(logits.dtype)
    return torch.where(row_has_donor, weights, torch.zeros_like(weights))


def explicit_spatial_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    allowed: Tensor,
    geometry_bias: Tensor,
    donor_measure: Tensor,
    measure_reference: Tensor,
    temperature: Tensor,
) -> tuple[Tensor, Tensor]:
    """Simple reference for the complete normalized-QK attention law."""
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("query/key/value must have shape [B,heads,N,width].")
    if allowed.shape != (query.shape[0], query.shape[2], key.shape[2]):
        raise ValueError("allowed has the wrong destination/donor axes.")
    q = F.normalize(query, dim=-1, eps=1.0e-6)
    k = F.normalize(key, dim=-1, eps=1.0e-6)
    logits = torch.matmul(q * temperature[None, :, None, None], k.transpose(-2, -1))
    logits = logits + geometry_bias
    measure = donor_measure.to(logits.dtype)
    safe_measure = torch.where(measure > 0, measure, torch.ones_like(measure))
    ref = measure_reference.to(logits.dtype).reshape(logits.shape[0], 1)
    log_measure = torch.log(safe_measure) - torch.log(ref.clamp_min(torch.finfo(logits.dtype).tiny))
    logits = logits + log_measure[:, None, None, :]
    expanded = allowed[:, None, :, :]
    weights = safe_masked_softmax(logits, expanded, dim=-1)
    return torch.matmul(weights, value), weights


class SpatialRouteAttentionV2(nn.Module):
    """One SDPA route with normalized Q/K and live bounded head temperatures."""

    def __init__(self, width: int, heads: int, coordinate_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("context width must be divisible by the attention head count.")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.coordinate_dim = int(coordinate_dim)
        self.dropout = float(dropout)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        score_input = 2 * coordinate_dim + 1
        self.geometry_score = nn.Sequential(
            nn.Linear(score_input, 32), nn.SiLU(), nn.Linear(32, heads, bias=False)
        )
        self.raw_temperature = nn.Parameter(torch.full((heads,), _initial_temperature_logit(self.head_width)))

    @property
    def temperature(self) -> Tensor:
        return bounded_temperature(self.raw_temperature)

    def forward(
        self,
        destination: Tensor,
        donor: Tensor,
        destination_coordinates: Tensor,
        donor_coordinates: Tensor,
        destination_lengths: Tensor,
        donor_lengths: Tensor,
        donor_measure: Tensor,
        measure_reference: Tensor,
        donor_mask: Tensor,
        destination_mask: Tensor,
        frame_scale: Tensor,
        *,
        remove_self: bool = False,
        return_attention: bool = False,
        geometry_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        batch, query_count, _ = destination.shape
        key_count = donor.shape[1]
        if key_count == 0:
            attention = destination.new_zeros(batch, self.heads, query_count, 0) if return_attention else None
            return destination.new_zeros(batch, query_count, self.width), attention

        q = self.query(self.query_norm(destination))
        donor_normalized = self.key_norm(donor)
        k = self.key(donor_normalized)
        v = self.value(donor_normalized)
        q = q.view(batch, query_count, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, key_count, self.heads, self.head_width).transpose(1, 2)
        v = v.view(batch, key_count, self.heads, self.head_width).transpose(1, 2)
        q = F.normalize(q, dim=-1, eps=1.0e-6)
        k = F.normalize(k, dim=-1, eps=1.0e-6)
        # Scaling Q by a live tensor and setting SDPA scale=1 avoids the
        # implementation's default 1/sqrt(head_width) a second time.
        q = q * self.temperature[None, :, None, None]

        if geometry_features is None:
            geometry_features, _ = pair_score_features(
                destination_coordinates,
                donor_coordinates,
                frame_scale,
                destination_lengths,
                donor_lengths,
            )
        geometry_bias = self.geometry_score(geometry_features).permute(0, 3, 1, 2)
        measure = donor_measure.to(dtype=destination.dtype)
        valid_donors = donor_mask & (measure > 0)
        allowed = valid_donors[:, None, :].expand(batch, query_count, key_count)
        allowed = allowed & destination_mask[:, :, None]
        if remove_self:
            if query_count != key_count:
                raise ValueError("Self-exclusion requires equal destination and donor axes.")
            allowed = allowed & ~torch.eye(query_count, dtype=torch.bool, device=destination.device)[None, :, :]

        safe_measure = torch.where(valid_donors, measure, torch.ones_like(measure))
        reference = measure_reference.to(dtype=destination.dtype).reshape(batch, 1)
        log_measure = torch.log(safe_measure) - torch.log(reference.clamp_min(torch.finfo(destination.dtype).tiny))
        additive = geometry_bias + log_measure[:, None, None, :]
        additive = additive.masked_fill(~allowed[:, None, :, :], -torch.inf)
        row_has_donor = allowed.any(dim=-1, keepdim=True)
        additive = torch.where(row_has_donor[:, None, :, :], additive, torch.zeros_like(additive))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=1.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_count, self.width)
        update = self.output(attended)
        valid_rows = row_has_donor & destination_mask[:, :, None]
        update = update * valid_rows.to(update.dtype)
        if not return_attention:
            return update, None
        # Trace-only explicit weights; the production path remains SDPA.
        logits = torch.matmul(q, k.transpose(-2, -1)) + additive
        weights = safe_masked_softmax(logits, allowed[:, None, :, :], dim=-1)
        return update, weights


class TokenFeedForwardV2(nn.Module):
    """One ordinary tokenwise pre-normalized residual FFN."""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        updated = values + self.dropout(self.net(self.norm(values)))
        return updated * mask[:, :, None].to(updated.dtype)


class StagedContextCycleV2(nn.Module):
    """MM → EM(new M) → EE(new E) → ME(new E), followed by one/type FFN."""

    def __init__(self, width: int, heads: int, coordinate_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mm = SpatialRouteAttentionV2(width, heads, coordinate_dim, dropout)
        self.em = SpatialRouteAttentionV2(width, heads, coordinate_dim, dropout)
        self.ee = SpatialRouteAttentionV2(width, heads, coordinate_dim, dropout)
        self.me = SpatialRouteAttentionV2(width, heads, coordinate_dim, dropout)
        self.environment_ff = TokenFeedForwardV2(width, dropout)
        self.module_ff = TokenFeedForwardV2(width, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        module_states: Tensor,
        environment_states: Tensor,
        scene,
        frame_scale: Tensor,
        *,
        return_attention: bool = False,
        geometry_features: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        module_lengths = scene.module_characteristic_lengths
        environment_lengths = scene.environment_characteristic_lengths
        module_measure = scene.module_measure
        environment_measure = scene.environment_measure
        module_reference = scene.module_measure_reference
        environment_reference = scene.environment_measure_reference
        batch = module_states.shape[0]
        if module_reference is None:
            module_reference = module_measure.new_ones(batch)
        if environment_reference is None:
            environment_reference = environment_measure.new_ones(batch)

        mm, mm_weights = self.mm(
            module_states, module_states,
            scene.module_coordinates, scene.module_coordinates,
            module_lengths, module_lengths,
            module_measure, module_reference,
            scene.module_mask, scene.module_mask, frame_scale,
            remove_self=True, return_attention=return_attention,
            geometry_features=None if geometry_features is None else geometry_features["mm"],
        )
        module_after_mm = module_states + self.dropout(mm)
        module_after_mm = module_after_mm * scene.module_mask[:, :, None].to(module_after_mm.dtype)

        em, em_weights = self.em(
            environment_states, module_after_mm,
            scene.environment_coordinates, scene.module_coordinates,
            environment_lengths, module_lengths,
            module_measure, module_reference,
            scene.module_mask, scene.environment_mask, frame_scale,
            return_attention=return_attention,
            geometry_features=None if geometry_features is None else geometry_features["em"],
        )
        environment_after_em = environment_states + self.dropout(em)
        environment_after_em = environment_after_em * scene.environment_mask[:, :, None].to(
            environment_after_em.dtype
        )

        ee, ee_weights = self.ee(
            environment_after_em, environment_after_em,
            scene.environment_coordinates, scene.environment_coordinates,
            environment_lengths, environment_lengths,
            environment_measure, environment_reference,
            scene.environment_mask, scene.environment_mask, frame_scale,
            return_attention=return_attention,
            geometry_features=None if geometry_features is None else geometry_features["ee"],
        )
        environment_after_ee = environment_after_em + self.dropout(ee)
        environment_next = self.environment_ff(environment_after_ee, scene.environment_mask)

        me, me_weights = self.me(
            module_after_mm, environment_next,
            scene.module_coordinates, scene.environment_coordinates,
            module_lengths, environment_lengths,
            environment_measure, environment_reference,
            scene.environment_mask, scene.module_mask, frame_scale,
            return_attention=return_attention,
            geometry_features=None if geometry_features is None else geometry_features["me"],
        )
        module_after_me = module_after_mm + self.dropout(me)
        module_next = self.module_ff(module_after_me, scene.module_mask)
        trace = {
            "module_after_mm": module_after_mm,
            "environment_after_em": environment_after_em,
            "environment_after_ee": environment_next,
            "module_after_me": module_next,
        }
        if return_attention:
            for name, value in (
                ("mm_attention", mm_weights),
                ("em_attention", em_weights),
                ("ee_attention", ee_weights),
                ("me_attention", me_weights),
            ):
                if value is not None:
                    trace[name] = value
        return module_next, environment_next, trace
