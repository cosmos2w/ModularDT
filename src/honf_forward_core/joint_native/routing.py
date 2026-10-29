"""Typed one-seed dispatch and the trained direct sparse control."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..native_retrieval.contracts import _require
from ..native_retrieval.seed_dispatch import SeedDispatchPlan
from .contracts import CanonicalGeometry, JointReadPlan, TypedNativeRecords


class ScoreCalibration(nn.Module):
    """Independent convex head mixture and bounded positive scalar calibration."""

    def __init__(self, heads: int, *, bias: bool = True, dtype=torch.float32):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive.")
        self.head_logits = nn.Parameter(torch.zeros(heads, dtype=dtype))
        self.raw_scale = nn.Parameter(torch.tensor(math.log(.2 / .8), dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros((), dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def scale(self) -> Tensor:
        return .25 + 3.75 * self.raw_scale.sigmoid()

    @property
    def head_weights(self) -> Tensor:
        return self.head_logits.softmax(0)

    def head_inputs(self, scores: Tensor, allowed: Tensor) -> Tensor:
        """Calibrated per-head inputs using the same eligible row maxima."""
        if scores.ndim != 4 or allowed.shape != (scores.shape[0], scores.shape[2], scores.shape[3]):
            raise ValueError("Scores/eligibility must have [B,H,Q,N]/[B,Q,N] shapes.")
        if scores.shape[1] != self.head_logits.numel():
            raise ValueError("Independent route head counts must match calibration.")
        _require(~allowed[:, None] | torch.isfinite(scores), "Eligible scores must be finite.")
        if not scores.shape[-1]:
            return scores
        maximum = scores.masked_fill(~allowed[:, None], -torch.inf).amax(-1, keepdim=True)
        maximum = torch.where(allowed[:, None].any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
        relative = torch.where(allowed[:, None], scores - maximum, torch.zeros_like(scores))
        result = relative * self.scale.to(scores)
        if self.bias is not None:
            result = result + self.bias.to(scores)
        return result

    def forward(self, scores: Tensor, allowed: Tensor) -> Tensor:
        if scores.ndim != 4 or allowed.shape != (scores.shape[0], scores.shape[2], scores.shape[3]):
            raise ValueError("Scores/eligibility must have [B,H,Q,N]/[B,Q,N] shapes.")
        if scores.shape[1] != self.head_logits.numel():
            raise ValueError("Independent route head counts must match calibration.")
        _require(~allowed[:, None] | torch.isfinite(scores), "Eligible scores must be finite.")
        if not scores.shape[-1]:
            return scores.sum(1)
        maximum = scores.masked_fill(~allowed[:, None], -torch.inf).amax(-1, keepdim=True)
        maximum = torch.where(allowed[:, None].any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
        relative = torch.where(allowed[:, None], scores - maximum, torch.zeros_like(scores))
        mixed = (relative * self.head_weights.to(scores)[None, :, None, None]).sum(1)
        result = mixed * self.scale.to(scores)
        if self.bias is not None:
            result = result + self.bias.to(scores)
        return result


def canonical_winner(scores: Tensor, allowed: Tensor, geometry: CanonicalGeometry):
    """Exact geometric ties in O(BQN); duplicate groups are scene-only.

    Counts are scattered into precomputed scene duplicate groups. No query by
    seed by seed tensor is constructed, including scenes containing duplicates.
    """
    if scores.ndim != 3 or scores.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("Canonical selection requires scores and boolean [B,Q,N] eligibility.")
    batch, queries, count = scores.shape
    if geometry.ranks.shape != (batch, count):
        raise ValueError("Canonical keys must match the native source axis.")
    if not count:
        return (torch.full((batch, queries), -1, device=scores.device, dtype=torch.long),
                allowed.clone(), torch.zeros((batch, queries), device=scores.device, dtype=torch.bool))
    candidate = scores.masked_fill(~allowed, -torch.inf)
    tied = (candidate == candidate.amax(-1, keepdim=True)) & allowed
    rank = geometry.ranks[:, None].expand(batch, queries, count)
    winner = torch.where(tied, rank, torch.full_like(rank, count)).argmin(-1)
    groups = geometry.duplicate_groups[:, None].expand(batch, queries, count)
    counts = torch.zeros((batch, queries, count), device=scores.device, dtype=torch.int32)
    counts.scatter_add_(2, groups, tied.to(torch.int32))
    unresolved = counts.amax(-1) > 1
    valid = allowed.any(-1) & ~unresolved
    return torch.where(valid, winner, torch.full_like(winner, -1)), tied, unresolved


def binary_sample(logits: Tensor, allowed: Tensor, mandatory: Tensor, *, deterministic=False,
                  straight_through=False, temperature=1., noise_scale=1., logistic_noise=None,
                  generator=None):
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("Sampling temperature/noise must be finite and positive/nonnegative.")
    if logistic_noise is not None and logistic_noise.shape != logits.shape:
        raise ValueError("Logistic noise must match the route's logits.")
    if deterministic:
        noisy = logits
    else:
        if logistic_noise is None:
            eps = torch.finfo(logits.dtype).eps
            uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype,
                                 generator=generator).clamp(eps, 1-eps)
            logistic_noise = uniform.log() - torch.log1p(-uniform)
        _require(torch.isfinite(logistic_noise), "Logistic noise must be finite.")
        noisy = logits + noise_scale * logistic_noise
    hard = ((noisy >= 0) & allowed) | mandatory
    probabilities = torch.where(mandatory, torch.ones_like(logits), logits.sigmoid() * allowed)
    relaxed = torch.where(mandatory, torch.ones_like(logits), (noisy / temperature).sigmoid() * allowed)
    gate = hard.to(logits) + (relaxed - relaxed.detach()) if straight_through else hard.to(logits)
    return hard, gate, probabilities


class JointDispatchParameters(nn.Module):
    def __init__(self, heads: int, *, dtype=torch.float32):
        super().__init__()
        self.module = ScoreCalibration(heads, bias=False, dtype=dtype)
        self.environment = ScoreCalibration(heads, bias=False, dtype=dtype)
        self.environment_offset = nn.Parameter(torch.zeros((), dtype=dtype))


def typed_query_logits(raw_M: Tensor, raw_E: Tensor, records: TypedNativeRecords,
                       parameters: JointDispatchParameters, query_mask: Tensor) -> Tensor:
    batch, _, queries, modules = raw_M.shape
    if modules != records.module_count or raw_E.shape[0] != batch or raw_E.shape[2] != queries:
        raise ValueError("Pre-E raw score axes must match the typed native catalogue.")
    allowed_M = query_mask[:, :, None] & records.module_present[:, None]
    allowed_E = query_mask[:, :, None] & records.environment_present[:, None]
    return torch.cat((parameters.module(raw_M, allowed_M),
                      parameters.environment(raw_E, allowed_E) + parameters.environment_offset.to(raw_E)), -1)


def dispatch_joint_native_seed(raw_M: Tensor, raw_E: Tensor, records: TypedNativeRecords,
                               parameters: JointDispatchParameters, *, query_mask=None,
                               deterministic=False, straight_through=False, temperature=1.,
                               noise_scale=1., gumbel_noise=None, generator=None) -> SeedDispatchPlan:
    records.assert_current()
    batch, _, queries, _ = raw_M.shape
    count = records.ids.shape[-1]
    if query_mask is None:
        query_mask = torch.ones((batch, queries), device=raw_M.device, dtype=torch.bool)
    if query_mask.shape != (batch, queries) or query_mask.dtype != torch.bool:
        raise ValueError("query_mask must be boolean [B,Q].")
    if temperature <= 0 or not math.isfinite(temperature) or noise_scale < 0 or not math.isfinite(noise_scale):
        raise ValueError("Invalid dispatch temperature/noise scale.")
    allowed = query_mask[:, :, None] & records.present[:, None]
    logits = typed_query_logits(raw_M, raw_E, records, parameters, query_mask).masked_fill(~allowed, -torch.inf)
    has_seed = allowed.any(-1, keepdim=True)
    safe = torch.where(has_seed, logits, torch.zeros_like(logits))
    probabilities = safe.softmax(-1) * allowed if count else logits.clone()
    if deterministic:
        noisy = logits
    else:
        if gumbel_noise is None:
            eps = torch.finfo(logits.dtype).eps
            uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype,
                                 generator=generator).clamp(eps, 1-eps)
            gumbel_noise = -(-uniform.log()).log()
        if gumbel_noise.shape != logits.shape:
            raise ValueError("Gumbel noise must be [B,Q,M+E].")
        _require(torch.isfinite(gumbel_noise), "Gumbel noise must be finite.")
        noisy = logits + noise_scale * gumbel_noise
    selected, tied, unresolved = canonical_winner(noisy, allowed, records.geometry)
    hard = torch.zeros_like(logits)
    if count:
        hard.scatter_(2, selected.clamp_min(0)[..., None], (selected >= 0).to(logits)[..., None])
    relaxed = torch.where(has_seed, noisy / temperature, torch.zeros_like(noisy))
    relaxed = relaxed.softmax(-1) * allowed if count else logits.clone()
    gate = hard + (relaxed - relaxed.detach()) if straight_through else hard
    selected_ids = records.ids.gather(1, selected.clamp_min(0)) if count else selected.clone()
    selected_ids = torch.where(selected >= 0, selected_ids, torch.full_like(selected_ids, -1))
    return SeedDispatchPlan(logits, probabilities, hard, gate, selected, selected_ids, tied,
                            unresolved, records.geometry.ranks)


def nearest_environment_protection(records: TypedNativeRecords, query_coordinates: Tensor,
                                   *, query_mask=None) -> Tensor:
    """Nearest positive-measure E under its original length-normalized metric."""
    batch, queries, _ = query_coordinates.shape
    environments = records.environment_ids.shape[1]
    if query_mask is None:
        query_mask = torch.ones((batch, queries), device=query_coordinates.device, dtype=torch.bool)
    allowed = (records.environment_present & (records.environment_measure > 0))[:, None] & query_mask[..., None]
    distance = ((query_coordinates[:, :, None] - records.environment_coordinates[:, None])
                / records.environment_lengths[:, None]).square().sum(-1)
    geometry = CanonicalGeometry.build(records.geometry.keys[:, records.module_count:])
    selected, _, unresolved = canonical_winner(-distance, allowed, geometry)
    _require(~unresolved, "Indistinguishable nearest E records require an explicit tied-union reference.")
    mask = torch.zeros((batch, queries, environments), dtype=torch.bool, device=query_coordinates.device)
    if environments:
        mask.scatter_(2, selected.clamp_min(0)[..., None], (selected >= 0)[..., None])
    return mask


class DirectGateParameters(nn.Module):
    def __init__(self, heads: int, *, dtype=torch.float32):
        super().__init__()
        self.module = ScoreCalibration(heads, dtype=dtype)
        self.environment = ScoreCalibration(heads, dtype=dtype)


def direct_typed_gates(raw_M: Tensor, raw_E: Tensor, records: TypedNativeRecords,
                       parameters: DirectGateParameters, protected_mask: Tensor, *, query_mask=None,
                       deterministic=False, straight_through=False, temperature=1., noise_scale=1.,
                       logistic_noise=None, generators=None) -> JointReadPlan:
    """S9 direct per-query Bernoulli gates with a typed best-donor floor."""
    records.assert_current()
    batch, _, queries, _ = raw_M.shape
    if query_mask is None:
        query_mask = torch.ones((batch, queries), device=raw_M.device, dtype=torch.bool)
    m = records.module_count
    if protected_mask.shape != (batch, queries, records.ids.shape[1]) or protected_mask.dtype != torch.bool:
        raise ValueError("Direct protections must be boolean [B,Q,M+E].")
    _require(~protected_mask | records.eligible[:, None], "Protections require eligible original records.")
    hard_parts, gates, floors, probabilities = [], [], [], []
    for name, raw, start, stop, calibration in (("module", raw_M, 0, m, parameters.module),
                                               ("environment", raw_E, m, records.ids.shape[1], parameters.environment)):
        allowed = query_mask[..., None] & records.eligible[:, None, start:stop]
        logits = calibration(raw, allowed)
        geometry = CanonicalGeometry.build(records.geometry.keys[:, start:stop])
        winner, _, unresolved = canonical_winner(logits, allowed, geometry)
        _require(~unresolved, "Indistinguishable direct donor ties require an explicit diagnostic.")
        floor = torch.zeros_like(allowed)
        if stop > start:
            floor.scatter_(2, winner.clamp_min(0)[..., None], (winner >= 0)[..., None])
        floor |= protected_mask[..., start:stop] & query_mask[..., None]
        hard, gate, probability = binary_sample(logits, allowed, floor, deterministic=deterministic,
            straight_through=straight_through, temperature=temperature, noise_scale=noise_scale,
            logistic_noise=(None if logistic_noise is None else
                            logistic_noise[..., start:stop] if isinstance(logistic_noise, Tensor) else logistic_noise[name]),
            generator=None if generators is None else generators[name])
        hard_parts.append(hard); gates.append(gate); floors.append(floor); probabilities.append(probability)
    return JointReadPlan(torch.cat(hard_parts, -1), torch.cat(gates, -1), torch.cat(floors, -1),
                         protected_mask & query_mask[..., None], m, probabilities=torch.cat(probabilities, -1))
