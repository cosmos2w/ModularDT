"""Input-only descriptions and risk selection for measured frontier actions.

The action is the realized hard source permission, not merely a receiver-tree
cut. This module deliberately has no work regressor: callers supply exact
canonical work measured from each candidate plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_interaction_cover import CaseLocalReceiverTree, MechanismPlan
from .input_cover_organizer import OrganizerScores
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class TypedActionSources:
    """One typed mechanism's pre-interaction inputs and realized hard action."""

    source_features: torch.Tensor  # [S,H], input-only source embeddings
    source_coordinates: torch.Tensor  # [S,D]
    source_measure: torch.Tensor  # [S], canonical source quadrature/weight
    permission: torch.Tensor  # [K,S], realized hard packet-source access
    score: torch.Tensor  # [K,S], input-only scorer output


def _typed_packet_summary(
    action: TypedActionSources,
    packet_coordinates: torch.Tensor,
    *,
    feature_dim: int,
    coordinate_dim: int,
) -> torch.Tensor:
    features, coordinates, measure, access, score = (
        action.source_features,
        action.source_coordinates,
        action.source_measure,
        action.permission,
        action.score,
    )
    packets = packet_coordinates.shape[0]
    sources = features.shape[0]
    if (features.shape != (sources, feature_dim)
            or coordinates.shape != (sources, coordinate_dim)
            or measure.shape != (sources,)
            or access.shape != (packets, sources)
            or score.shape != (packets, sources)):
        raise ValueError("Typed action source and packet axes do not align.")
    if (not all(torch.isfinite(value).all() for value in (features, coordinates, measure, access, score))
            or torch.any(measure < 0) or torch.any(access < 0) or torch.any(access > 1)):
        raise ValueError("Typed action inputs must be finite with nonnegative measures and access in [0,1].")
    measure = measure.to(device=features.device, dtype=features.dtype)
    included = measure[None] * access
    excluded = measure[None] * (1 - access)
    included_mass = included.sum(dim=1, keepdim=True)
    excluded_mass = excluded.sum(dim=1, keepdim=True)
    included_features = included @ features / included_mass.clamp_min(1e-8)
    excluded_features = excluded @ features / excluded_mass.clamp_min(1e-8)
    relative = coordinates[None] - packet_coordinates[:, None]
    included_relative = (included[..., None] * relative).sum(dim=1) / included_mass.clamp_min(1e-8)
    excluded_relative = (excluded[..., None] * relative).sum(dim=1) / excluded_mass.clamp_min(1e-8)
    included_spread = (
        included * relative.square().sum(dim=-1)
    ).sum(dim=1, keepdim=True) / included_mass.clamp_min(1e-8)
    source_count = (measure > 0).to(features.dtype).sum().expand(packets, 1)
    degree = ((access > 0) & (measure > 0)[None]).to(features.dtype).sum(dim=1, keepdim=True)
    included_score = (included * score).sum(dim=1, keepdim=True) / included_mass.clamp_min(1e-8)
    excluded_score = (excluded * score).sum(dim=1, keepdim=True) / excluded_mass.clamp_min(1e-8)
    return torch.cat((
        included_features, excluded_features,
        included_relative, excluded_relative,
        included_mass.log1p(), excluded_mass.log1p(), included_spread.sqrt(),
        source_count.log1p(), degree.log1p(), included_score, excluded_score,
    ), dim=-1)


def describe_frontier_action(
    packet_features: torch.Tensor,
    packet_coordinates: torch.Tensor,
    typed_actions: Mapping[str, TypedActionSources],
    *,
    mechanism_order: tuple[str, ...] = ("MM", "QE"),
) -> torch.Tensor:
    """Return packet rows sensitive to selected and discarded source sets.

    Permuting packets permutes output rows; permuting sources with all their
    metadata leaves the output unchanged. Missing mechanisms are rejected so
    an inapplicable route cannot silently masquerade as a trained action.
    """

    if packet_features.ndim != 2 or packet_coordinates.ndim != 2:
        raise ValueError("Packet features and coordinates must be two-dimensional.")
    packets, width = packet_features.shape
    if packets < 1 or packet_coordinates.shape[0] != packets:
        raise ValueError("At least one packet and aligned coordinates are required.")
    if not torch.isfinite(packet_features).all() or not torch.isfinite(packet_coordinates).all():
        raise ValueError("Packet inputs must be finite.")
    if set(typed_actions) != set(mechanism_order):
        raise ValueError("Typed action mechanisms must match the declared order exactly.")
    summaries = [packet_features]
    for mechanism in mechanism_order:
        summaries.append(_typed_packet_summary(
            typed_actions[mechanism], packet_coordinates,
            feature_dim=width, coordinate_dim=packet_coordinates.shape[1],
        ))
    return torch.cat(summaries, dim=-1)


def describe_realized_plan(
    scores: OrganizerScores,
    plan: MechanismPlan,
    encoded: EncodedInterfaceCase,
    cut: tuple[int, ...],
    *,
    case_index: int = 0,
) -> torch.Tensor:
    """Adapt an actual G hard plan into packet features without target fields."""

    tree = plan.tree
    if not cut or min(cut) < 0 or max(cut) >= len(tree.nodes):
        raise ValueError("Realized cut contains an unavailable receiver node.")
    if scores.node_embeddings is None or scores.module_embeddings is None or scores.environment_embeddings is None:
        raise ValueError("Action description requires the frozen input-only score embeddings.")
    cut_rows = torch.as_tensor(cut, device=scores.node_embeddings.device, dtype=torch.long)
    node_features = scores.node_embeddings[cut_rows]
    universe = tree.universe
    packet_centers = []
    for node in cut:
        indices = list(tree.nodes[node].anchor_indices)
        weights = universe.weights[indices]
        packet_centers.append((
            universe.coordinates[indices] * weights[:, None]
        ).sum(dim=0) / weights.sum().clamp_min(1e-8))
    centers = torch.stack(packet_centers)
    if case_index < 0 or case_index >= encoded.module_centers.shape[0]:
        raise IndexError("Action case index is outside the encoded batch.")
    typed = {}
    for mechanism, source_features, coordinates, measure in (
        ("MM", scores.module_embeddings, encoded.module_centers[case_index], encoded.module_present[case_index]),
        ("QE", scores.environment_embeddings, encoded.env_coords[case_index], encoded.env_weights[case_index]),
    ):
        if mechanism not in scores.mechanism_logits:
            raise ValueError(f"Missing frozen {mechanism} source scores for action description.")
        typed[mechanism] = TypedActionSources(
            source_features=source_features,
            source_coordinates=coordinates,
            source_measure=measure,
            permission=plan.permission_matrix(mechanism)[cut_rows].detach(),
            score=scores.mechanism_logits[mechanism][cut_rows].detach(),
        )
    return describe_frontier_action(node_features, centers, typed)


def receiver_role_descriptors(tree: CaseLocalReceiverTree, role_count: int) -> torch.Tensor:
    """Input-only geometry/count descriptors for each physical receiver role."""

    if role_count < 1:
        raise ValueError("At least one receiver role is required.")
    universe = tree.universe
    coordinates = universe.coordinates / universe.coordinate_scale
    result = []
    for role in range(role_count):
        mask = universe.roles == role
        if bool(mask.any()):
            weight = universe.weights[mask]
            total = weight.sum()
            mean = (coordinates[mask] * weight[:, None]).sum(dim=0) / total
            variance = ((coordinates[mask] - mean).square() * weight[:, None]).sum(dim=0) / total
            count = coordinates.new_tensor([float(mask.sum())]).log1p()
            mass = total.reshape(1).log1p()
        else:
            mean = coordinates.new_zeros((coordinates.shape[1],))
            variance = mean
            count = coordinates.new_zeros((1,))
            mass = count
        one_hot = F.one_hot(torch.tensor(role, device=coordinates.device), role_count).to(coordinates.dtype)
        result.append(torch.cat((mean, variance.sqrt(), count, mass, one_hot)))
    return torch.stack(result)


class ActionAwareRiskHead(nn.Module):
    """Small permutation-invariant predictor of signed log risk for each role."""

    def __init__(
        self, *, packet_feature_dim: int, budget_dim: int,
        receiver_role_dim: int, hidden_dim: int = 24,
    ) -> None:
        super().__init__()
        if min(packet_feature_dim, budget_dim, receiver_role_dim, hidden_dim) < 1:
            raise ValueError("Action-risk dimensions must be positive.")
        self.packet_feature_dim = int(packet_feature_dim)
        self.budget_dim = int(budget_dim)
        self.receiver_role_dim = int(receiver_role_dim)
        self.packet_encoder = nn.Sequential(
            nn.LayerNorm(packet_feature_dim), nn.Linear(packet_feature_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.risk = nn.Sequential(
            nn.Linear(3 * hidden_dim + budget_dim + receiver_role_dim + 1, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim // 2 or 1),
            nn.SiLU(), nn.Linear(hidden_dim // 2 or 1, 1),
        )

    def forward(
        self, packet_rows: torch.Tensor, budget_vector: torch.Tensor,
        receiver_role_features: torch.Tensor,
        *, nonredundant_k: int | None = None,
    ) -> torch.Tensor:
        if packet_rows.ndim != 2 or packet_rows.shape[1] != self.packet_feature_dim or packet_rows.shape[0] < 1:
            raise ValueError("Action packet rows have the wrong shape.")
        if budget_vector.shape != (self.budget_dim,):
            raise ValueError("Action budget has the wrong shape.")
        if receiver_role_features.ndim != 2 or receiver_role_features.shape[1] != self.receiver_role_dim:
            raise ValueError("Per-role receiver descriptors have the wrong shape.")
        if not all(torch.isfinite(value).all() for value in (packet_rows, budget_vector, receiver_role_features)):
            raise ValueError("Action-risk inputs must be finite.")
        effective_k = packet_rows.shape[0] if nonredundant_k is None else int(nonredundant_k)
        if not 1 <= effective_k <= packet_rows.shape[0]:
            raise ValueError("Nonredundant packet count must fit the realized action cut.")
        encoded = self.packet_encoder(packet_rows)
        pooled = torch.cat((
            encoded.mean(dim=0), encoded.sum(dim=0), encoded.amax(dim=0),
            packet_rows.new_tensor([math.log1p(effective_k)]),
            budget_vector,
        ))
        common = pooled[None].expand(receiver_role_features.shape[0], -1)
        return self.risk(torch.cat((common, receiver_role_features), dim=-1)).squeeze(-1)


def signed_log_role_risk(
    candidate_error: torch.Tensor, incumbent_error: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """Signed log error ratio to the fixed incumbent, including improvements."""

    if candidate_error.shape != incumbent_error.shape or candidate_error.shape != epsilon.shape:
        raise ValueError("Candidate, incumbent, and numerical floors must align.")
    if (not all(torch.isfinite(value).all() for value in (candidate_error, incumbent_error, epsilon))
            or torch.any(candidate_error < 0) or torch.any(incumbent_error < 0)
            or torch.any(epsilon <= 0)):
        raise ValueError("Risk errors must be nonnegative and floors positive finite.")
    return torch.log((candidate_error + epsilon) / (incumbent_error + epsilon))


def incumbent_role_log_limit(
    incumbent_error: torch.Tensor,
    numerical_floor: torch.Tensor,
    absolute_allowance: torch.Tensor,
    *,
    relative_allowance: float = 0.10,
) -> torch.Tensor:
    """Map a fixed-incumbent physical-error gate into signed-log risk units.

    The numerical floor stabilizes the risk label; it is distinct from the
    explicitly justified absolute physical-error allowance in the gate.
    """

    if (incumbent_error.shape != numerical_floor.shape
            or incumbent_error.shape != absolute_allowance.shape):
        raise ValueError("Incumbent error, numerical floor, and allowance must align.")
    if (not all(torch.isfinite(value).all() for value in (
        incumbent_error, numerical_floor, absolute_allowance
    )) or torch.any(incumbent_error < 0) or torch.any(numerical_floor <= 0)
            or torch.any(absolute_allowance < 0)
            or not math.isfinite(relative_allowance) or relative_allowance < 0):
        raise ValueError("Incumbent error and gate allowances must be finite and nonnegative.")
    allowed_error = (1.0 + relative_allowance) * incumbent_error + absolute_allowance
    return torch.log((allowed_error + numerical_floor) / (incumbent_error + numerical_floor))


def action_risk_loss(
    predicted: torch.Tensor, measured: torch.Tensor,
    *, ranking_weight: float = 0.1,
) -> torch.Tensor:
    """Robust role regression with a small within-case action-ranking term."""

    if predicted.shape != measured.shape or predicted.ndim != 2:
        raise ValueError("Prediction and measured action risk must align as [actions,roles].")
    if not all(torch.isfinite(value).all() for value in (predicted, measured)):
        raise ValueError("Action risks must be finite.")
    if ranking_weight < 0:
        raise ValueError("Ranking weight cannot be negative.")
    regression = F.smooth_l1_loss(predicted, measured)
    if predicted.shape[0] < 2 or ranking_weight == 0:
        return regression
    measured_scalar = measured.amax(dim=1)
    predicted_scalar = predicted.amax(dim=1)
    row, column = torch.triu_indices(predicted.shape[0], predicted.shape[0], offset=1, device=predicted.device)
    difference = measured_scalar[row] - measured_scalar[column]
    informative = difference.abs() > 1e-3
    if not bool(informative.any()):
        return regression
    signed_gap = difference[informative].sign() * (
        predicted_scalar[row[informative]] - predicted_scalar[column[informative]]
    )
    return regression + ranking_weight * F.softplus(-signed_gap).mean()


@dataclass(frozen=True)
class ActionRiskSelection:
    index: int
    unsupported_at_budget: bool
    safe_sparse_indices: tuple[int, ...]


def select_action_by_risk(
    predicted_log_risk: torch.Tensor,
    exact_work: torch.Tensor,
    packet_counts: torch.Tensor,
    trained_sparse: torch.Tensor,
    role_log_limits: torch.Tensor,
    *,
    full_access_index: int,
    empirical_margin: torch.Tensor | None = None,
) -> ActionRiskSelection:
    """Choose least exact work among predicted-safe sparse actions or explicit full access."""

    actions, roles = predicted_log_risk.shape
    if (actions < 1 or not 0 <= full_access_index < actions
            or exact_work.shape != (actions,) or packet_counts.shape != (actions,)
            or trained_sparse.shape != (actions,)
            or role_log_limits.shape != (roles,)):
        raise ValueError("Action selection arrays have incompatible dimensions.")
    margin = torch.zeros_like(role_log_limits) if empirical_margin is None else empirical_margin
    if margin.shape != (roles,) or not all(torch.isfinite(value).all() for value in (
        predicted_log_risk, exact_work, packet_counts, role_log_limits, margin
    )):
        raise ValueError("Action selection inputs must be finite and aligned.")
    if torch.any(exact_work < 0) or torch.any(packet_counts < 1) or torch.any(margin < 0):
        raise ValueError("Work, packet counts, and empirical margins must be nonnegative.")
    safe = ((predicted_log_risk + margin[None]) <= role_log_limits[None]).all(dim=1)
    candidates = [
        index for index in range(actions)
        if index != full_access_index and bool(trained_sparse[index]) and bool(safe[index])
    ]
    if not candidates:
        return ActionRiskSelection(full_access_index, True, ())
    chosen = min(candidates, key=lambda index: (
        float(exact_work[index]), int(packet_counts[index]), index,
    ))
    return ActionRiskSelection(chosen, False, tuple(candidates))


__all__ = [
    "ActionAwareRiskHead", "ActionRiskSelection", "TypedActionSources",
    "action_risk_loss", "describe_frontier_action", "describe_realized_plan",
    "incumbent_role_log_limit", "receiver_role_descriptors", "select_action_by_risk",
    "signed_log_role_risk",
]
