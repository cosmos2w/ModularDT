"""Continuous exact-zero retrieval and measure-aware zero-message reduction."""
from __future__ import annotations
import torch


def positive_quadratic_gate(x):
    """C1 psi(x), with no overflow when a finite positive input is squared."""
    positive = x.clamp_min(0)
    small = positive.clamp_max(1)
    reciprocal = positive.clamp_min(1).reciprocal()
    return torch.where(positive <= 1, small.square() / (1 + small.square()),
                       1 / (1 + reciprocal.square()))


def entmax15(scores, mask=None):
    """Exact sort/threshold 1.5-entmax; ties receive equal mass; empty rows zero."""
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if scores.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError('Entmax mask must be boolean with the score shape.')
    if not scores.shape[-1]:
        return scores.clone()
    any_valid = mask.any(-1, keepdim=True)
    maximum = scores.masked_fill(~mask, -torch.inf).amax(-1, keepdim=True)
    maximum = torch.where(any_valid, maximum, 0)
    # A bounded sentinel keeps masked sorted-prefix moments finite. It is never
    # used as a valid donor, and lies strictly below the possible support.
    shifted = torch.where(mask, scores - maximum, 0) / 2
    minimum = shifted.masked_fill(~mask, torch.inf).amin(-1,keepdim=True)
    minimum = torch.where(any_valid,minimum,0)
    shifted = torch.where(mask,shifted,minimum-2)
    ordered = shifted.sort(-1, descending=True).values
    ranks = torch.arange(1, scores.shape[-1]+1, device=scores.device, dtype=scores.dtype)
    mean = ordered.cumsum(-1) / ranks
    mean_square = ordered.square().cumsum(-1) / ranks
    variance_sum = ranks * (mean_square - mean.square())
    threshold = mean - ((1 - variance_sum) / ranks).clamp_min(torch.finfo(scores.dtype).tiny).sqrt()
    support_size = ((threshold <= ordered) & (ranks <= mask.sum(-1, keepdim=True))).sum(-1, keepdim=True).clamp_min(1)
    tau = threshold.gather(-1, support_size-1)
    result = (shifted-tau).clamp_min(0).square() * mask
    return torch.where(any_valid, result, 0)


def guarded_weights(scores, gates, measure, valid, query_mask, epsilon=1e-3, reference=None):
    """Return [B,Q,N,H] coefficients and [B,Q,H] reduced message mass.

    The all-eligible score normalizer remains a declared control-plane path.
    Zero-measure physical M sources are excluded only from information weights.
    """
    if epsilon != 1e-3:
        raise ValueError('continuous_joint_native_v1 fixes epsilon=1e-3.')
    if not scores.shape[-1]:
        shape = (scores.shape[0],scores.shape[2],0,scores.shape[1])
        return scores.new_zeros(shape), scores.new_zeros(shape[:2]+shape[3:])
    eligible = valid & (measure > 0)
    allowed = eligible[:,None,None,:] & query_mask[:,None,:,None]
    # Preserve the literal parent's raw measure arithmetic on ordinary inputs.
    # In particular, at g=1 the SAME raw tensor is reduced for both R and S,
    # giving Z=1 bitwise rather than summing already-rounded normalized weights.
    tiny=torch.finfo(scores.dtype).tiny
    reference=measure.new_ones(measure.shape[0]) if reference is None else reference.to(measure)
    base_reference=reference.clamp_min(tiny)
    physical_measure=torch.where(eligible,measure,torch.zeros_like(measure))
    largest=physical_measure.amax(-1)
    # A detached range check prevents overflow before either branch is built;
    # selecting a log-space fallback after inf arithmetic poisons its VJP.
    ratio=largest.detach()/base_reference.detach()
    ordinary=torch.isfinite(ratio)&(ratio<=torch.finfo(scores.dtype).max/(2*measure.shape[-1]))
    safe_reference=torch.where(ordinary,base_reference,largest.clamp_min(tiny))
    relative_measure=physical_measure/safe_reference[:,None]
    maximum=scores.masked_fill(~allowed,-torch.inf).amax(-1,keepdim=True)
    maximum=torch.where(allowed.any(-1,keepdim=True),maximum,0)
    raw=torch.exp(torch.where(allowed,scores-maximum,-torch.inf))*relative_measure[:,None,None,:]
    total=raw.sum(-1,keepdim=True)
    usable=torch.isfinite(total)&(total>0)
    safe_raw=torch.where(usable,raw,torch.zeros_like(raw))
    safe_total=torch.where(usable,total,torch.ones_like(total))
    weighted=safe_raw*gates[:,None]
    z=weighted.sum(-1,keepdim=True)/safe_total
    denominator=z+epsilon*(1-z).square()
    coefficient=(weighted/safe_total)/denominator
    mass=z/denominator

    # Underflow can make every raw product zero despite eligible donors. Use
    # the stable log route only on those rows; empty query/type rows stay zero.
    fallback_allowed=allowed & ~usable
    log_measure = torch.where(eligible, measure, torch.ones_like(measure)).log()
    logits = scores + log_measure[:,None,None,:]
    maximum = logits.masked_fill(~fallback_allowed,-torch.inf).amax(-1,keepdim=True)
    maximum = torch.where(fallback_allowed.any(-1,keepdim=True),maximum,0)
    weight = torch.exp(torch.where(fallback_allowed,logits-maximum,-torch.inf))
    w = weight / weight.sum(-1,keepdim=True).clamp_min(tiny)
    u = w * gates[:,None]
    z = u.sum(-1,keepdim=True)
    denominator = z + epsilon*(1-z).square()
    coefficient=torch.where(usable,coefficient,u/denominator)
    mass=torch.where(usable,mass,z/denominator)
    return coefficient.permute(0,2,3,1), mass.squeeze(-1).transpose(1,2)


def support_surrogate(gates, delta):
    return (1+delta)*gates/(gates+delta)
