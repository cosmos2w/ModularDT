"""Measure-aware soft and sparse normalizations for typed IAG atlases."""

from __future__ import annotations

import torch
from torch import Tensor


def _check_shapes(
    logits: Tensor, donor_measure: Tensor, donor_valid: Tensor, destination_valid: Tensor
) -> None:
    if logits.ndim != 4:
        raise ValueError("Atlas logits must have shape [B,H,E,N].")
    batch, _heads, anchors, donors = logits.shape
    if donor_measure.shape != (batch, donors):
        raise ValueError("donor_measure must have shape [B,N].")
    if donor_valid.shape != (batch, donors) or donor_valid.dtype != torch.bool:
        raise ValueError("donor_valid must be boolean [B,N].")
    if destination_valid.shape != (batch, anchors) or destination_valid.dtype != torch.bool:
        raise ValueError("destination_valid must be boolean [B,E].")


def _calculation_dtype(logits: Tensor, measure: Tensor) -> torch.dtype:
    if logits.dtype == torch.float64 or measure.dtype == torch.float64:
        return torch.float64
    return torch.float32


def normalize_donor_measure(donor_measure: Tensor, donor_valid: Tensor) -> Tensor:
    """Normalize each scene's positive present donor measure, retaining gradients."""
    if donor_measure.ndim != 2 or donor_valid.shape != donor_measure.shape:
        raise ValueError("donor_measure and donor_valid must both have shape [B,N].")
    if donor_valid.dtype != torch.bool:
        raise TypeError("donor_valid must be boolean.")
    dtype = torch.float64 if donor_measure.dtype == torch.float64 else torch.float32
    measure = donor_measure.to(dtype=dtype)
    malformed = donor_valid & (~torch.isfinite(measure) | (measure < 0))
    if malformed.any():
        raise ValueError("Present donor measures must be finite and nonnegative.")
    valid = donor_valid & (measure > 0)
    positive_measure = torch.where(valid, measure, torch.zeros_like(measure))
    total = positive_measure.sum(dim=-1, keepdim=True)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    return positive_measure / safe_total


def _normalized_inputs(
    logits: Tensor,
    donor_measure: Tensor,
    donor_valid: Tensor,
    destination_valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _check_shapes(logits, donor_measure, donor_valid, destination_valid)
    dtype = _calculation_dtype(logits, donor_measure)
    scores = logits.to(dtype=dtype)
    measure = donor_measure.to(dtype=dtype)
    valid_donor = donor_valid & (measure > 0)
    normalized_measure = normalize_donor_measure(measure, donor_valid)
    allowed = valid_donor[:, None, None, :] & destination_valid[:, None, :, None]
    allowed = allowed.expand_as(scores)
    return scores, normalized_measure, valid_donor, allowed, destination_valid


def measure_relative_softmax(
    logits: Tensor,
    donor_measure: Tensor,
    donor_valid: Tensor,
    destination_valid: Tensor,
) -> Tensor:
    """Return ``m_i exp(s_i)`` normalized over each typed donor row.

    The measure is normalized over valid donors per scene. Its common scale
    cancels in softmax, but keeping the normalized value in this calculation
    makes the physical reference measure explicit and reusable by transport.
    Empty donor types and invalid destinations return exact zeros.
    """
    scores, normalized_measure, valid_donor, allowed, _ = _normalized_inputs(
        logits, donor_measure, donor_valid, destination_valid
    )
    safe_measure = torch.where(valid_donor, normalized_measure, torch.ones_like(normalized_measure))
    log_measure = torch.log(safe_measure)[:, None, None, :]
    weighted = (scores + log_measure).masked_fill(~allowed, -torch.inf)
    row_has_donor = allowed.any(dim=-1, keepdim=True)
    safe_weighted = torch.where(row_has_donor, weighted, torch.zeros_like(weighted))
    probabilities = torch.softmax(safe_weighted, dim=-1)
    probabilities = probabilities * allowed.to(probabilities.dtype)
    return probabilities.to(dtype=logits.dtype)


def measure_relative_entmax15_reference(
    logits: Tensor, donor_measure: Tensor, donor_valid: Tensor | None = None
) -> Tensor:
    """FP64 scalar-row bisection reference for the measure-relative 1.5 law.

    This deliberately favors a transparent monotone root solve over production
    throughput. ``logits`` and ``donor_measure`` must be one-dimensional.
    """
    if logits.ndim != 1 or donor_measure.shape != logits.shape:
        raise ValueError("The reference solver accepts one-dimensional score and measure rows.")
    if donor_valid is None:
        donor_valid = donor_measure > 0
    if donor_valid.shape != logits.shape or donor_valid.dtype != torch.bool:
        raise ValueError("donor_valid must be boolean and match the score row.")
    scores = logits.to(torch.float64)
    measure = donor_measure.to(torch.float64)
    valid = donor_valid & (measure > 0)
    safe_measure = torch.where(valid, measure, torch.zeros_like(measure))
    total = safe_measure.sum()
    if not bool(total > 0):
        return torch.zeros_like(scores)
    normalized_measure = safe_measure / total
    live_scores = scores[valid]
    live_measure = normalized_measure[valid]
    # At the lower endpoint, the smallest-measure donor alone contributes at
    # least one unit of mass; at max(score), every contribution is zero.
    low = live_scores.min() - 2.0 / torch.sqrt(live_measure.min())
    high = live_scores.max()
    for _ in range(128):
        middle = (low + high) * 0.5
        mass = (live_measure * torch.relu((live_scores - middle) * 0.5).square()).sum()
        if bool(mass > 1.0):
            low = middle
        else:
            high = middle
    tau = (low + high) * 0.5
    result = torch.zeros_like(scores)
    result[valid] = normalized_measure[valid] * torch.relu((scores[valid] - tau) * 0.5).square()
    return result


def measure_relative_entmax15(
    logits: Tensor,
    donor_measure: Tensor,
    donor_valid: Tensor,
    destination_valid: Tensor,
) -> Tensor:
    """Vectorized sorted-prefix solution of the measure-relative 1.5 law.

    Prefix moments and thresholds use FP64: FP32 cancellation can leave a gap
    between adjacent candidate intervals at a donor's support boundary. Select
    the last prefix whose final donor is above its threshold, as in the usual
    entmax support test; a second rounded interval comparison is unnecessary.
    The root retains score and measure gradients. Outputs retain the input
    dtype, and there is no loop over anchors or queries.
    """
    scores, normalized_measure, _valid_donor, allowed, _ = _normalized_inputs(
        logits, donor_measure, donor_valid, destination_valid
    )
    batch, heads, anchors, donors = scores.shape
    if donors == 0:
        return torch.zeros_like(logits)
    scores = scores.to(torch.float64)
    root_measure = normalized_measure.to(torch.float64)
    row_has_donor = allowed.any(dim=-1, keepdim=True)
    masked_scores = scores.masked_fill(~allowed, -torch.inf)
    row_max = masked_scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(row_has_donor, row_max, torch.zeros_like(row_max))
    shifted_scores = torch.where(allowed, scores - row_max, -torch.inf)
    sorted_scores, order = torch.sort(shifted_scores, dim=-1, descending=True, stable=True)
    measure_rows = root_measure[:, None, None, :].expand(batch, heads, anchors, donors)
    sorted_measure = torch.gather(measure_rows, -1, order)
    live_scores = torch.where(torch.isfinite(sorted_scores), sorted_scores, torch.zeros_like(sorted_scores))
    prefix_a = torch.cumsum(sorted_measure, dim=-1)
    prefix_b = torch.cumsum(sorted_measure * live_scores, dim=-1)
    prefix_c = torch.cumsum(sorted_measure * live_scores.square(), dim=-1)
    discriminant = prefix_b.square() - prefix_a * (prefix_c - 4.0)
    prefix_eligible = (prefix_a > 0) & row_has_donor & torch.isfinite(sorted_scores)
    positive_discriminant = discriminant > 0
    safe_discriminant = torch.where(
        prefix_eligible & positive_discriminant, discriminant, torch.ones_like(discriminant)
    )
    sqrt_discriminant = torch.sqrt(safe_discriminant)
    safe_a = torch.where(prefix_a > 0, prefix_a, torch.ones_like(prefix_a))
    candidate_tau = (prefix_b - sqrt_discriminant) / safe_a
    candidate_ok = (sorted_scores > candidate_tau) & prefix_eligible & positive_discriminant
    has_root = candidate_ok.any(dim=-1, keepdim=True)
    root_or_empty = has_root | ~row_has_donor
    if logits.device.type == "cuda":
        torch._assert_async(root_or_empty.all(), "The measure-relative entmax15 prefix root was not found.")
    elif not bool(root_or_empty.all()):
        raise RuntimeError("The measure-relative entmax15 prefix root was not found.")
    prefix_index = torch.arange(donors, device=logits.device)
    root_index = torch.where(candidate_ok, prefix_index, 0).amax(dim=-1, keepdim=True)
    tau = torch.gather(candidate_tau, -1, root_index)
    tau = torch.where(row_has_donor, tau, torch.zeros_like(tau))
    probabilities = root_measure[:, None, None, :] * torch.relu(
        (scores - row_max - tau) * 0.5
    ).square()
    probabilities = probabilities * allowed.to(probabilities.dtype)
    return probabilities.to(dtype=logits.dtype)


def measure_relative_normalize(
    logits: Tensor,
    donor_measure: Tensor,
    donor_valid: Tensor,
    destination_valid: Tensor,
    *,
    normalization: str,
) -> tuple[Tensor, Tensor]:
    """Select a supported normalization and return probabilities plus measure."""
    normalized_measure = normalize_donor_measure(donor_measure, donor_valid)
    if normalization == "softmax":
        probabilities = measure_relative_softmax(
            logits, donor_measure, donor_valid, destination_valid
        )
    elif normalization == "entmax15":
        probabilities = measure_relative_entmax15(
            logits, donor_measure, donor_valid, destination_valid
        )
    else:
        raise ValueError("normalization must be 'softmax' or 'entmax15'.")
    return probabilities, normalized_measure
