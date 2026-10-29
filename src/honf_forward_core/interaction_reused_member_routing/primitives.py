"""Numerical primitives used by the anchored member reader."""
from __future__ import annotations

import torch
from torch import Tensor

def prior_weighted_scores(scores: Tensor, prior: Tensor, allowed: Tensor) -> Tensor:
    """Center and bound scores using the candidate-normalized relation prior.

    A unit variance floor keeps the bounded score scale stable. A member with
    vanishing prior mass therefore has vanishing influence on the calibration
    statistics, even while it remains relation-eligible.
    """
    if scores.shape != prior.shape or scores.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("Member scores, candidate prior and mask must have equal shapes.")
    weights = prior * allowed.to(prior.dtype)
    total = weights.sum(-1, keepdim=True)
    weights = torch.where(total > 0, weights / total.clamp_min(torch.finfo(weights.dtype).tiny),
                          torch.zeros_like(weights))
    mean = (weights * scores).sum(-1, keepdim=True)
    centered = torch.where(allowed, scores - mean, torch.zeros_like(scores))
    variance = (weights * centered.square()).sum(-1, keepdim=True)
    scaled = centered / (4. * torch.sqrt(variance + 1.))
    return 4. * torch.tanh(scaled) * allowed.to(scores.dtype)

def build_joint_member_measure(
    module_mask: Tensor,
    environment_mask: Tensor,
    environment_measure: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return μ with unit total measure for each present M/E role.

    M records use a discrete equal measure. Eligible E records use their
    positive physical cell measures normalized within E. Empty roles remain
    all zero and do not borrow mass from the other role.
    """
    if module_mask.ndim != 2 or environment_mask.ndim != 2 or environment_measure.shape != environment_mask.shape:
        raise ValueError("M/E masks and E measure must have compatible [B,N] shapes.")
    if module_mask.shape[0] != environment_mask.shape[0]:
        raise ValueError("M and E member batches must agree.")
    if module_mask.dtype != torch.bool or environment_mask.dtype != torch.bool:
        raise TypeError("M/E member masks must be boolean.")
    dtype = dtype or environment_measure.dtype
    m_mask = module_mask
    e_measure = environment_measure.to(dtype)
    e_mask = environment_mask & (e_measure > 0)
    m_count = m_mask.sum(-1, keepdim=True).to(dtype).clamp_min(1.)
    e_total = torch.where(e_mask, e_measure, torch.zeros_like(e_measure)).sum(-1, keepdim=True)
    mu_m = m_mask.to(dtype) / m_count
    mu_e = torch.where(e_mask, e_measure / e_total.clamp_min(torch.finfo(dtype).tiny),
                       torch.zeros_like(e_measure))
    return torch.cat((mu_m, mu_e), dim=-1)

def read_probability(contrast: Tensor) -> Tensor:
    """Softsign Bernoulli probability, using stable rational tail identities.

    Its real-arithmetic derivative is 1/(2*(1+d*d)**1.5), rather than an
    exact-zero support derivative. Native loss sensitivity is still required.
    """
    radius = torch.hypot(contrast, torch.ones_like(contrast))
    # Both branches remain finite, including the unused one. Signed clamps
    # preserve the correct derivative 1/2 at zero (abs would give zero there).
    negative_tail = 0.5 / radius / (radius - contrast.clamp_max(0.))
    positive_tail = 0.5 / radius / (radius + contrast.clamp_min(0.))
    return torch.where(contrast < 0, negative_tail, 1. - positive_tail)
