"""Balanced losses for hierarchical inverse rectified flows.

Request ``R`` and context ``c`` condition flow matching for compact plan ``G``
and layout ``D``. Joint losses compare frozen-HONF realized ``G_hat`` and
request functionals. Flow terms remain primary so consistency cannot collapse
the intended one-to-many generators into deterministic optimization.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def compact_plan_validity_loss(continuous_endpoint: torch.Tensor) -> torch.Tensor:
    """Soft endpoint penalties for the independent normalized plan state."""

    coords = continuous_endpoint[..., 0:4]
    masses = continuous_endpoint[..., 4:6]
    scales = continuous_endpoint[..., 6:8]
    fractions = continuous_endpoint[..., 8:10]
    bounds = F.relu(-coords).square().mean() + F.relu(coords - 1.0).square().mean()
    nonnegative = F.relu(-masses).square().mean() + F.relu(-scales).square().mean() + F.relu(-fractions).square().mean()
    simplex = (masses.sum(dim=1) - 1.0).square().mean() + (fractions.sum(dim=1) - 1.0).square().mean()
    return bounds + nonnegative + simplex


def plan_training_losses(
    *,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    activity_logits: torch.Tensor,
    activity_target: torch.Tensor,
    endpoint_estimate: torch.Tensor,
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    selected = {"flow": 1.0, "activity": 0.10, "validity": 0.05, **dict(weights or {})}
    flow = F.mse_loss(predicted_velocity, target_velocity)
    activity = F.binary_cross_entropy_with_logits(activity_logits, activity_target.float())
    validity = compact_plan_validity_loss(endpoint_estimate)
    auxiliary = selected["activity"] * activity + selected["validity"] * validity
    total = selected["flow"] * flow + auxiliary
    return {"total": total, "flow": flow, "activity": activity, "validity": validity, "auxiliary": auxiliary}


def layout_geometry_loss(
    endpoint: torch.Tensor,
    present: torch.Tensor,
    geometry_constraints: torch.Tensor,
) -> torch.Tensor:
    """Differentiable normalized center/clearance surrogate used in stages 2-4."""

    centers = endpoint[..., :2]
    mask = present.float()
    inlet = geometry_constraints[:, 4:5]
    outlet = geometry_constraints[:, 5:6]
    wall = geometry_constraints[:, 3:4]
    edge_penalty = (
        F.relu(inlet - centers[..., 0]).square()
        + F.relu(centers[..., 0] - (1.0 - outlet)).square()
        + F.relu(wall - centers[..., 1]).square()
        + F.relu(centers[..., 1] - (1.0 - wall)).square()
    )
    edge = _weighted_mean(edge_penalty, mask)
    delta = centers[:, :, None, :] - centers[:, None, :, :]
    distance = torch.sqrt(delta.square().sum(dim=-1) + 1.0e-8) / (2.0**0.5)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    pair_mask = pair_mask * (1.0 - torch.eye(centers.shape[1], device=centers.device).unsqueeze(0))
    minimum = geometry_constraints[:, 2:3, None]
    pair = _weighted_mean(F.relu(minimum - distance).square(), pair_mask)
    return edge + pair


def layout_plan_alignment_loss(
    endpoint: torch.Tensor,
    present: torch.Tensor,
    compact_plan: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Align generated module centroids/allocation with compact-plan sources.

    The soft nearest-source assignment is only a training auxiliary. Exact
    ``G_hat`` remains owned by frozen HONF in stage four and evaluation.
    """

    centers = endpoint[..., :2]
    sources = compact_plan[..., 1:3]
    active_edges = compact_plan[..., 0].float()
    delta = centers[:, :, None, :] - sources[:, None, :, :]
    logits = -delta.square().sum(dim=-1) / max(float(temperature), 1.0e-6)
    logits = logits.masked_fill(active_edges[:, None, :] < 0.5, -1.0e4)
    assignment = torch.softmax(logits, dim=-1) * present.float().unsqueeze(-1)
    edge_mass = assignment.sum(dim=1)
    centroids = torch.einsum("bmk,bmd->bkd", assignment, centers)
    centroids = centroids / edge_mass.unsqueeze(-1).clamp_min(1.0e-6)
    source_error = (centroids - sources).square().mean(dim=-1)
    source = _weighted_mean(source_error, active_edges)
    realized_fraction = edge_mass / edge_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_fraction = compact_plan[..., 11]
    allocation = _weighted_mean((realized_fraction - target_fraction).square(), active_edges)
    return source + allocation


def layout_training_losses(
    *,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    presence_logits: torch.Tensor,
    presence_target: torch.Tensor,
    count_logits: torch.Tensor,
    count_target: torch.Tensor,
    endpoint_estimate: torch.Tensor,
    layout_target: torch.Tensor,
    geometry_constraints: torch.Tensor,
    compact_plan: torch.Tensor,
    inactive_flow_weight: float = 0.25,
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    selected = {
        "flow": 1.0, "presence": 0.10, "count": 0.05,
        "geometry": 0.05, "heat_range": 0.02, "plan_alignment": 0.05,
        **dict(weights or {}),
    }
    slot_weight = inactive_flow_weight + (1.0 - inactive_flow_weight) * presence_target.float()
    flow = _weighted_mean((predicted_velocity - target_velocity).square().mean(dim=-1), slot_weight)
    presence = F.binary_cross_entropy_with_logits(presence_logits, presence_target.float())
    count = F.cross_entropy(count_logits, count_target.long())
    geometry = layout_geometry_loss(endpoint_estimate, presence_target, geometry_constraints)
    predicted_total = (endpoint_estimate[..., 2] * presence_target).sum(dim=1)
    target_total = (layout_target[..., 2] * presence_target).sum(dim=1)
    heat_range = F.mse_loss(predicted_total, target_total)
    plan_alignment = layout_plan_alignment_loss(endpoint_estimate, presence_target, compact_plan)
    auxiliary = (
        selected["presence"] * presence
        + selected["count"] * count
        + selected["geometry"] * geometry
        + selected["heat_range"] * heat_range
        + selected["plan_alignment"] * plan_alignment
    )
    total = selected["flow"] * flow + auxiliary
    return {
        "total": total, "flow": flow, "presence": presence, "count": count,
        "geometry": geometry, "heat_range": heat_range, "auxiliary": auxiliary,
        "plan_alignment": plan_alignment,
    }


def joint_consistency_losses(
    *,
    request_loss: torch.Tensor,
    plan_distance: torch.Tensor,
    geometry_loss: torch.Tensor,
    correction_magnitude: torch.Tensor | None = None,
    correction_target_loss: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
    flow_reference: torch.Tensor | None = None,
    max_consistency_to_flow_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Combine exact/smooth joint terms with an explicit flow-dominance cap."""

    selected = {
        "request": 0.10, "plan": 0.10, "geometry": 0.05,
        "correction": 0.02, "correction_target": 0.10, **dict(weights or {}),
    }
    correction = request_loss.new_zeros(()) if correction_magnitude is None else correction_magnitude.mean()
    correction_target = (
        request_loss.new_zeros(()) if correction_target_loss is None else correction_target_loss
    )
    raw = (
        selected["request"] * request_loss
        + selected["plan"] * plan_distance
        + selected["geometry"] * geometry_loss
        + selected["correction"] * correction
        + selected["correction_target"] * correction_target
    )
    if flow_reference is not None:
        cap = flow_reference.detach().clamp_min(1.0e-6) * float(max_consistency_to_flow_ratio)
        scale = torch.clamp(cap / raw.detach().clamp_min(1.0e-8), max=1.0)
        consistency = raw * scale
    else:
        consistency = raw
    return {
        "total": consistency,
        "request": request_loss,
        "plan": plan_distance,
        "geometry": geometry_loss,
        "correction": correction,
        "correction_target": correction_target,
        "raw_weighted": raw,
    }


__all__ = [
    "compact_plan_validity_loss",
    "joint_consistency_losses",
    "layout_geometry_loss",
    "layout_plan_alignment_loss",
    "layout_training_losses",
    "plan_training_losses",
]
