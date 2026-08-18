"""Sparse probability transforms shared by organizer and field routing."""

from __future__ import annotations

from typing import Optional

import torch


EPS = 1e-12


def locality_bias(
    radius_square: torch.Tensor,
    *,
    mode: str,
    strength: float,
    radius_cap: float,
) -> torch.Tensor:
    """Convert normalized squared distance into a routing-logit bias."""

    if mode == "bounded_gaussian":
        cap_square = radius_square.new_tensor(float(radius_cap) ** 2)
        return -0.5 * float(strength) * torch.minimum(radius_square, cap_square)
    if mode == "gaussian_bounded":
        cap = radius_square.new_tensor(float(radius_cap))
        return -0.5 * float(strength) * torch.clamp(radius_square, max=cap)
    if mode == "compact_kernel":
        compactness = torch.relu(1.0 - radius_square).square()
        return float(strength) * torch.log(compactness + 1.0e-6)
    if mode == "none":
        return torch.zeros_like(radius_square)
    raise ValueError(f"Unsupported locality mode: {mode!r}.")


def entmax15(
    logits: torch.Tensor,
    *,
    dim: int = -1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply exact 1.5-entmax with optional zero-mass masked entries.

    Rows whose mask contains no valid entry return all zeros. Half and bfloat16
    inputs use float32 working arithmetic and are cast back on return.
    """

    if logits.numel() == 0:
        return logits
    working_dtype = torch.float32 if logits.dtype in {torch.float16, torch.bfloat16} else logits.dtype
    values = logits.to(dtype=working_dtype)
    valid: Optional[torch.Tensor]
    if mask is None:
        valid = None
    else:
        valid = torch.broadcast_to(mask.to(device=logits.device, dtype=torch.bool), logits.shape)
        values = values.masked_fill(~valid, -1.0e9)
    values = values / 2.0
    values = values - values.amax(dim=dim, keepdim=True)
    if valid is not None:
        values = values.masked_fill(~valid, -1.0e4)

    sorted_values, sorted_indices = torch.sort(values, dim=dim, descending=True)
    dimension = values.shape[dim]
    rho_shape = [1] * values.ndim
    rho_shape[dim] = dimension
    rho = torch.arange(1, dimension + 1, device=values.device, dtype=values.dtype).view(rho_shape)
    mean = sorted_values.cumsum(dim=dim) / rho
    mean_square = sorted_values.square().cumsum(dim=dim) / rho
    variance_sum = rho * (mean_square - mean.square())
    delta = (1.0 - variance_sum) / rho
    taus = mean - torch.sqrt(torch.relu(delta))
    support = taus <= sorted_values
    if valid is not None:
        sorted_valid = torch.gather(valid, dim, sorted_indices)
        support = support & sorted_valid
    support_size = support.sum(dim=dim, keepdim=True)
    tau_index = (support_size - 1).clamp_min(0)
    tau_star = torch.gather(taus, dim, tau_index)
    probabilities = torch.relu(values - tau_star).square()
    if valid is not None:
        probabilities = probabilities * valid.to(dtype=probabilities.dtype)
    mass = probabilities.sum(dim=dim, keepdim=True)
    probabilities = torch.where(mass > 0, probabilities / mass.clamp_min(EPS), probabilities)
    return probabilities.to(dtype=logits.dtype)


def normalize_assignment(
    logits: torch.Tensor,
    *,
    mode: str,
    dim: int = -1,
    mask: Optional[torch.Tensor] = None,
    entmax_blend: Optional[float] = None,
) -> torch.Tensor:
    """Apply a strict dense-softmax or exact-sparse entmax normalization."""

    if mode == "entmax15":
        return entmax15(logits, dim=dim, mask=mask)
    if mode not in {"softmax", "scheduled"}:
        raise ValueError(f"Unsupported assignment normalizer: {mode!r}.")
    if mask is None:
        softmax_probabilities = torch.softmax(logits, dim=dim)
        valid = None
    else:
        valid = torch.broadcast_to(mask.to(device=logits.device, dtype=torch.bool), logits.shape)
        masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        softmax_probabilities = torch.softmax(masked_logits, dim=dim) * valid.to(dtype=logits.dtype)
        softmax_probabilities = softmax_probabilities / softmax_probabilities.sum(
            dim=dim, keepdim=True
        ).clamp_min(EPS)
    if mode == "softmax":
        return softmax_probabilities
    if entmax_blend is None:
        raise ValueError("scheduled assignment normalization requires entmax_blend.")
    blend = min(max(float(entmax_blend), 0.0), 1.0)
    if blend <= 0.0:
        return softmax_probabilities
    entmax_probabilities = entmax15(logits, dim=dim, mask=mask)
    if blend >= 1.0:
        return entmax_probabilities
    probabilities = (1.0 - blend) * softmax_probabilities + blend * entmax_probabilities
    if valid is not None:
        probabilities = probabilities * valid.to(dtype=probabilities.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(EPS)


def schedule_fraction(epoch: int, start_epoch: int, transition_epochs: int) -> float:
    """Return an exact piecewise-linear schedule fraction in ``[0,1]``."""

    if int(start_epoch) < 0:
        return 0.0
    if int(epoch) <= int(start_epoch):
        return 0.0
    if int(transition_epochs) <= 0:
        return 1.0
    if int(epoch) >= int(start_epoch) + int(transition_epochs):
        return 1.0
    return float(int(epoch) - int(start_epoch)) / float(int(transition_epochs))


__all__ = ["entmax15", "locality_bias", "normalize_assignment", "schedule_fraction"]
