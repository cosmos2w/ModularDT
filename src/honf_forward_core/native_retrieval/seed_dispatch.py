"""One-native-seed query dispatch with geometry-only deterministic tie handling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .contracts import tensor_versions


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def _bounded_scale(raw: Tensor) -> Tensor:
    return 0.25 + 3.75 * torch.sigmoid(raw)


def _check_tensor(condition: Tensor, message: str) -> None:
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _canonical_ranks(source_geometry: Tensor) -> Tensor:
    """Return lexicographic ranks over coordinates/lengths, independent of IDs."""
    batch, seeds, dimensions = source_geometry.shape
    if seeds == 0:
        return torch.empty((batch, 0), dtype=torch.long, device=source_geometry.device)
    order = torch.arange(seeds, dtype=torch.long, device=source_geometry.device).expand(batch, seeds)
    for dimension in range(dimensions - 1, -1, -1):
        values = source_geometry[:, :, dimension].gather(1, order)
        by_value = torch.argsort(values, dim=1, stable=True)
        order = order.gather(1, by_value)
    ranks = torch.empty_like(order)
    canonical_rank = torch.arange(seeds, dtype=torch.long, device=source_geometry.device).expand(batch, seeds)
    ranks.scatter_(1, order, canonical_rank)
    return ranks


def _relative_query_scores(raw_scores: Tensor, seed_valid: Tensor) -> Tensor:
    if raw_scores.ndim != 4:
        raise ValueError("raw_query_scores must have shape [B,H,Q,S].")
    batch, _, _, seeds = raw_scores.shape
    if seed_valid.shape != (batch, seeds) or seed_valid.dtype != torch.bool:
        raise ValueError("seed_valid must be boolean [B,S].")
    if seeds == 0:
        return torch.zeros_like(raw_scores)
    allowed = seed_valid[:, None, None, :]
    has_seed = allowed.any(dim=-1, keepdim=True)
    masked = raw_scores.masked_fill(~allowed, -torch.inf)
    maximum = masked.amax(dim=-1, keepdim=True)
    maximum = torch.where(has_seed, maximum, torch.zeros_like(maximum))
    relative = torch.where(allowed & has_seed, raw_scores - maximum, torch.zeros_like(raw_scores))
    _check_tensor(torch.isfinite(relative).all(), "Eligible query-to-seed scores must be finite.")
    return relative


class SeedDispatchParameters(nn.Module):
    """Convex head mix and bounded scale over original A2-D query scores."""

    def __init__(self, heads: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive.")
        self.head_logits = nn.Parameter(torch.zeros(heads, dtype=dtype))
        self.raw_scale = nn.Parameter(torch.tensor(_inverse_sigmoid(0.2), dtype=dtype))

    @property
    def head_weights(self) -> Tensor:
        return torch.softmax(self.head_logits, dim=0)

    @property
    def scale(self) -> Tensor:
        return _bounded_scale(self.raw_scale)


@dataclass(frozen=True)
class SeedDispatchPlan:
    """One seed axis index per query plus the estimator and tie diagnostics."""

    dispatch_logits: Tensor  # [B,Q,S], invalid scores are -inf
    probabilities: Tensor  # categorical softmax of unnoised calibrated scores
    hard: Tensor  # one-hot [B,Q,S], or all-zero for empty/indeterminate queries
    straight_through: Tensor  # hard forward, Gumbel-softmax backward when enabled
    dispatch_seed: Tensor  # long [B,Q], original module-axis index or -1
    dispatch_seed_ids: Tensor  # long [B,Q], labels for trace only or -1
    tied_seed_mask: Tensor  # bool [B,Q,S], exact top-score tie set
    tie_indeterminate: Tensor  # bool [B,Q], top tie has geometrically duplicate seeds
    canonical_rank: Tensor  # long [B,S]
    _tensor_versions: tuple[tuple[int, int, int], ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dispatch_logits.ndim != 3:
            raise ValueError("dispatch_logits must have shape [B,Q,S].")
        batch, queries, seeds = self.dispatch_logits.shape
        for name, value in (
            ("probabilities", self.probabilities),
            ("hard", self.hard),
            ("straight_through", self.straight_through),
            ("tied_seed_mask", self.tied_seed_mask),
        ):
            if value.shape != (batch, queries, seeds):
                raise ValueError(f"{name} must have shape [B,Q,S].")
        if self.tied_seed_mask.dtype != torch.bool:
            raise TypeError("tied_seed_mask must be boolean.")
        if self.dispatch_seed.shape != (batch, queries) or self.dispatch_seed.dtype not in (torch.int32, torch.int64):
            raise ValueError("dispatch_seed must be integer [B,Q].")
        if self.dispatch_seed_ids.shape != (batch, queries) or self.dispatch_seed_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("dispatch_seed_ids must be integer [B,Q].")
        if self.tie_indeterminate.shape != (batch, queries) or self.tie_indeterminate.dtype != torch.bool:
            raise ValueError("tie_indeterminate must be boolean [B,Q].")
        if self.canonical_rank.shape != (batch, seeds) or self.canonical_rank.dtype not in (torch.int32, torch.int64):
            raise ValueError("canonical_rank must be integer [B,S].")
        object.__setattr__(self, "_tensor_versions", tensor_versions(*self._owned_tensors()))

    def _owned_tensors(self) -> tuple[Tensor, ...]:
        return (
            self.dispatch_logits,
            self.probabilities,
            self.hard,
            self.straight_through,
            self.dispatch_seed,
            self.dispatch_seed_ids,
            self.tied_seed_mask,
            self.tie_indeterminate,
            self.canonical_rank,
        )

    def assert_current(self) -> None:
        if tensor_versions(*self._owned_tensors()) != self._tensor_versions:
            raise ValueError("Seed dispatch logits, probabilities, ties, or selected indices changed in place.")


def dispatch_native_seed(
    raw_query_scores: Tensor,
    seed_valid: Tensor,
    source_geometry: Tensor,
    parameters: SeedDispatchParameters,
    *,
    gumbel_noise: Tensor | None = None,
    noise_scale: float = 1.0,
    temperature: float = 1.0,
    deterministic: bool = False,
    straight_through: bool = False,
    query_mask: Tensor | None = None,
    seed_ids: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SeedDispatchPlan:
    """Choose one valid input-native seed using calibrated original query scores.

    Native IDs are retained only for correspondence and traces. Deterministic ties
    use lexicographic coordinates plus characteristic lengths; if top-scoring seeds
    are geometrically indistinguishable, the plan returns index -1 and exposes the
    full tied set so the caller must use a tied-union reference or reject the query.
    """
    if raw_query_scores.ndim != 4:
        raise ValueError("raw_query_scores must have shape [B,H,Q,S].")
    batch, heads, queries, seeds = raw_query_scores.shape
    if seed_valid.shape != (batch, seeds) or seed_valid.dtype != torch.bool:
        raise ValueError("seed_valid must be boolean [B,S].")
    if source_geometry.ndim != 3 or source_geometry.shape[:2] != (batch, seeds):
        raise ValueError("source_geometry must have shape [B,S,G].")
    if source_geometry.shape[-1] == 0:
        raise ValueError("source_geometry must contain coordinates or other physical geometry.")
    if parameters.head_logits.shape != (heads,):
        raise ValueError("Dispatch parameter and query score head counts differ.")
    if noise_scale < 0.0 or not math.isfinite(float(noise_scale)):
        raise ValueError("noise_scale must be finite and nonnegative.")
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and positive.")
    if query_mask is None:
        query_mask = torch.ones((batch, queries), dtype=torch.bool, device=raw_query_scores.device)
    if query_mask.shape != (batch, queries) or query_mask.dtype != torch.bool:
        raise ValueError("query_mask must be boolean [B,Q].")
    if seed_ids is not None and (seed_ids.shape != (batch, seeds) or seed_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("seed_ids must be integer [B,S].")
    if parameters.head_logits.device != raw_query_scores.device:
        raise ValueError("Dispatch parameters and scores must share a device.")
    _check_tensor(torch.isfinite(source_geometry[seed_valid]).all(), "Valid seed geometry must be finite.")

    relative = _relative_query_scores(raw_query_scores, seed_valid)
    weights = parameters.head_weights.to(raw_query_scores)
    scores = parameters.scale.to(raw_query_scores) * (relative * weights[None, :, None, None]).sum(dim=1)
    scores = scores.masked_fill(~seed_valid[:, None, :], 0.0)
    valid_query_seed = query_mask[:, :, None] & seed_valid[:, None, :]
    dispatch_logits = scores.masked_fill(~valid_query_seed, -torch.inf)
    has_seed = valid_query_seed.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_seed, dispatch_logits, torch.zeros_like(dispatch_logits))
    probabilities = torch.softmax(safe_logits, dim=-1) * valid_query_seed.to(scores.dtype)
    probabilities = torch.where(has_seed, probabilities, torch.zeros_like(probabilities))

    if deterministic:
        candidate = dispatch_logits
    else:
        if gumbel_noise is None:
            tiny = torch.finfo(raw_query_scores.dtype).eps
            uniform = torch.rand(
                (batch, queries, seeds),
                device=raw_query_scores.device,
                dtype=raw_query_scores.dtype,
                generator=generator,
            ).clamp(min=tiny, max=1.0 - tiny)
            gumbel_noise = -torch.log(-torch.log(uniform))
        if gumbel_noise.shape != (batch, queries, seeds):
            raise ValueError("gumbel_noise must have shape [B,Q,S].")
        candidate = dispatch_logits + float(noise_scale) * gumbel_noise
    candidate = candidate.masked_fill(~valid_query_seed, -torch.inf)

    if seeds:
        best = candidate.amax(dim=-1, keepdim=True)
        has_valid = valid_query_seed.any(dim=-1)
        tied = (candidate == best) & valid_query_seed
        rank = _canonical_ranks(source_geometry)
        rank_for_query = rank[:, None, :].expand(batch, queries, seeds)
        winner = torch.where(tied, rank_for_query, torch.full_like(rank_for_query, seeds)).argmin(dim=-1)
        geometry_equal = (source_geometry[:, :, None, :] == source_geometry[:, None, :, :]).all(dim=-1)
        upper = torch.triu(torch.ones((seeds, seeds), dtype=torch.bool, device=raw_query_scores.device), diagonal=1)
        tied_pairs = tied[:, :, :, None] & tied[:, :, None, :] & geometry_equal[:, None, :, :] & upper[None, None, :, :]
        duplicate_tie = tied_pairs.any(dim=(-1, -2))
        unresolved = duplicate_tie & has_valid & query_mask
        selected = torch.where(has_valid & ~unresolved, winner, torch.full_like(winner, -1))
        hard = torch.zeros((batch, queries, seeds), dtype=raw_query_scores.dtype, device=raw_query_scores.device)
        safe_selected = selected.clamp(min=0)
        hard.scatter_(2, safe_selected.unsqueeze(-1), (selected >= 0).to(hard.dtype).unsqueeze(-1))
        hard = hard * (has_valid & ~unresolved & query_mask).unsqueeze(-1).to(hard.dtype)
        tied_seed_mask = tied
        tie_indeterminate = unresolved
        relaxed_logits = dispatch_logits if deterministic else dispatch_logits + float(noise_scale) * gumbel_noise
        relaxed_logits = relaxed_logits / float(temperature)
        relaxed_logits = relaxed_logits.masked_fill(~valid_query_seed, -torch.inf)
        safe_relaxed_logits = torch.where(has_seed, relaxed_logits, torch.zeros_like(relaxed_logits))
        relaxed = torch.softmax(safe_relaxed_logits, dim=-1) * valid_query_seed.to(hard.dtype)
        relaxed = torch.where(has_seed, relaxed, torch.zeros_like(relaxed))
        straight_through_tensor = hard + (relaxed - relaxed.detach()) if straight_through else hard
        if seed_ids is None:
            selected_ids = selected
        else:
            selected_ids = seed_ids.gather(1, safe_selected)
            selected_ids = torch.where(selected >= 0, selected_ids, torch.full_like(selected_ids, -1))
    else:
        rank = torch.empty((batch, 0), dtype=torch.long, device=raw_query_scores.device)
        selected = torch.full((batch, queries), -1, dtype=torch.long, device=raw_query_scores.device)
        selected_ids = selected.clone()
        hard = raw_query_scores.new_zeros((batch, queries, 0))
        straight_through_tensor = hard
        tied_seed_mask = torch.zeros((batch, queries, 0), dtype=torch.bool, device=raw_query_scores.device)
        tie_indeterminate = torch.zeros((batch, queries), dtype=torch.bool, device=raw_query_scores.device)

    return SeedDispatchPlan(
        dispatch_logits=dispatch_logits,
        probabilities=probabilities,
        hard=hard,
        straight_through=straight_through_tensor,
        dispatch_seed=selected,
        dispatch_seed_ids=selected_ids,
        tied_seed_mask=tied_seed_mask,
        tie_indeterminate=tie_indeterminate,
        canonical_rank=rank,
    )


__all__ = ["SeedDispatchParameters", "SeedDispatchPlan", "dispatch_native_seed"]
