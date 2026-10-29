"""Exact native-batch denominators for bounded microbatch physical AD."""

from __future__ import annotations

from typing import Any

import torch


def physical_denominator_masses(batch: dict[str, Any]) -> dict[str, float]:
    """Measure denominators from original native inputs, without a model call."""

    target = batch["field_targets"]
    weights = batch.get("point_weights")
    if weights is None:
        field = float(target.numel())
    else:
        weights = torch.as_tensor(weights, dtype=target.dtype, device=target.device)
        if weights.ndim == target.ndim and weights.shape[-1] == 1:
            weights = weights.squeeze(-1)
        while weights.ndim < target.ndim - 1:
            weights = weights.unsqueeze(-1)
        field = float(torch.broadcast_to(weights, target.shape[:-1]).sum()) * target.shape[-1]
    present = batch["structure"]["module_present"].float()
    valid = batch.get("interface_condition_valid_mask")
    h = present if valid is None else present[:, :, None] * valid.float()
    return {"field": field, "module": float(present.sum()), "valid_h": float(h.sum())}


def physical_normalization_weights(batch: dict[str, Any], native_masses: dict[str, float], case_fraction: float) -> dict[str, float]:
    """Scale local means so ordinary case-weighted backward sums native means.

    Module query/port widths and channels stay identical when slicing one
    already-collated native batch. Their constant factors cancel. Weighted
    query mass and valid-h masks have separate native denominators.
    """

    measured = physical_denominator_masses(batch)
    return {key: measured[key] / max(native_masses[key], 1e-6) / case_fraction for key in measured}


def native_denominators_enabled(campaign: dict[str, Any], epoch: int) -> bool:
    start = campaign.get("native_loss_denominators_start_epoch")
    return start is not None and epoch >= int(start)
