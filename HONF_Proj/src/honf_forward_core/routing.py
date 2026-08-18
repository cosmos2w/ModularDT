"""Sparse probability transforms shared by organizer and field routing."""

from __future__ import annotations

from typing import Optional

import torch


EPS = 1e-12


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
) -> torch.Tensor:
    """Apply a strict dense-softmax or exact-sparse entmax normalization."""

    if mode == "entmax15":
        return entmax15(logits, dim=dim, mask=mask)
    if mode != "softmax":
        raise ValueError(f"Unsupported assignment normalizer: {mode!r}.")
    if mask is None:
        return torch.softmax(logits, dim=dim)
    valid = torch.broadcast_to(mask.to(device=logits.device, dtype=torch.bool), logits.shape)
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked_logits, dim=dim) * valid.to(dtype=logits.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(EPS)


__all__ = ["entmax15", "normalize_assignment"]
