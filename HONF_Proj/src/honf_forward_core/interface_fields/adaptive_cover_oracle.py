"""Bounded train-only search for an adequate interaction cover.

The caller supplies frozen absolute-model observations on training probes.
This module never reads reference labels itself and never runs at deployment.
Its row counts are exact for the declared probe queries; complete cost uses
separately measured coefficients and is only a proposal estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import torch

from .adaptive_interaction_cover import AdaptiveCoverPlan, CoverPairLedger, compile_cover_pairs


@dataclass(frozen=True)
class RoleError:
    """One protected role's observed reference and frozen-teacher errors."""

    reference: float | None
    teacher: float
    resolved: bool


@dataclass(frozen=True)
class OracleObservation:
    roles: Mapping[str, RoleError]
    reference_evidence_id: str
    teacher_checkpoint_id: str


@dataclass(frozen=True)
class RoleLimit:
    absolute_error: float
    allowed_increase: float


@dataclass(frozen=True)
class MeasuredCostCoefficients:
    """Coefficients fit from real preparation/organization/execution timing."""

    measurement_id: str
    preparation_ms: float
    organization_ms_per_node: float  # per active source-bearing node, not tree capacity
    route_ms_per_path: float
    qm_ms_per_row: float
    qe_ms_per_row: float
    pack_ms_per_row: float
    wrapper_ms: float

    def __post_init__(self) -> None:
        if not self.measurement_id:
            raise ValueError("cost coefficients require a measured provenance ID")
        for key in (
            "preparation_ms", "organization_ms_per_node", "route_ms_per_path",
            "qm_ms_per_row", "qe_ms_per_row", "pack_ms_per_row", "wrapper_ms",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0.0:
                raise ValueError(f"cost coefficient {key} must be finite and nonnegative")

    def estimate_ms(self, plan: AdaptiveCoverPlan, ledger: CoverPairLedger) -> float:
        return (
            self.preparation_ms
            + self.organization_ms_per_node * plan.active_group_count()
            + self.route_ms_per_path * (ledger.qm_raw_paths + ledger.qe_raw_paths)
            + self.qm_ms_per_row * ledger.qm_unique_rows
            + self.qe_ms_per_row * ledger.qe_unique_rows
            + self.pack_ms_per_row * (ledger.qm_unique_rows + ledger.qe_unique_rows)
            + self.wrapper_ms
        )


@dataclass(frozen=True)
class OracleProposal:
    kind: str
    node: int
    source_type: str | None
    source_index: int | None
    plan: AdaptiveCoverPlan
    train_evidence_id: str


@dataclass(frozen=True)
class OracleTrial:
    proposal: OracleProposal
    observation: OracleObservation
    ledger: CoverPairLedger
    estimated_ms: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class OracleResult:
    baseline_observation: OracleObservation
    baseline_ledger: CoverPairLedger
    baseline_estimated_ms: float
    selected_plan: AdaptiveCoverPlan
    selected_observation: OracleObservation
    selected_ledger: CoverPairLedger
    selected_estimated_ms: float
    trials: tuple[OracleTrial, ...]


def _adequate(
    candidate: OracleObservation,
    baseline: OracleObservation,
    limits: Mapping[str, RoleLimit],
) -> tuple[bool, str]:
    for role, limit in limits.items():
        current = candidate.roles.get(role)
        reference = baseline.roles.get(role)
        if current is None or reference is None or not current.resolved or not reference.resolved:
            return False, f"unresolved_role:{role}"
        if current.reference is None or reference.reference is None:
            return False, f"missing_reference:{role}"
        if not (math.isfinite(current.reference) and math.isfinite(current.teacher)):
            return False, f"nonfinite_role:{role}"
        if current.reference > limit.absolute_error:
            return False, f"absolute_error:{role}"
        if current.reference > reference.reference + limit.allowed_increase:
            return False, f"relative_to_full:{role}"
    return True, "adequate"


def local_cover_proposals(
    plan: AdaptiveCoverPlan,
    *,
    module_relevance: torch.Tensor,
    environment_relevance: torch.Tensor,
    resolved_module: torch.Tensor,
    resolved_environment: torch.Tensor,
    relevance_threshold: float,
    train_evidence_id: str,
) -> tuple[OracleProposal, ...]:
    """Generate split/prune/merge proposals from frozen training evidence.

    Unknown source relevance is never pruned. A split proposal includes the
    jointly needed child pruning, since splitting two full-access children
    alone cannot reduce the fine-pair union.
    """

    if not train_evidence_id:
        raise ValueError("oracle proposals need a training-evidence identifier")
    if module_relevance.shape != plan.module_membership.shape or environment_relevance.shape != plan.environment_membership.shape:
        raise ValueError("relevance maps must align with typed candidate-source memberships")
    if resolved_module.shape != module_relevance.shape or resolved_environment.shape != environment_relevance.shape:
        raise ValueError("reliability masks must align with relevance maps")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance threshold must be in [0,1]")
    proposals: list[OracleProposal] = []
    active_nodes = (
        plan.tree.access(plan.tree.universe.coordinates, plan.split_gates) > 0
    ).any(dim=0)
    for node_id, node in enumerate(plan.tree.nodes):
        if not bool(active_nodes[node_id]):
            continue
        if not node.is_leaf:
            if float(plan.split_gates[node_id].detach()) == 0.0:
                candidate = plan.with_split(node_id, 1.0)
                module = candidate.module_membership.clone()
                environment = candidate.environment_membership.clone()
                for child in (node.left, node.right):
                    assert child is not None
                    module[child] = torch.where(
                        resolved_module[child] & (module_relevance[child] <= relevance_threshold),
                        torch.zeros_like(module[child]), module[child],
                    )
                    environment[child] = torch.where(
                        resolved_environment[child]
                        & (environment_relevance[child] <= relevance_threshold),
                        torch.zeros_like(environment[child]), environment[child],
                    )
                candidate = replace(candidate, module_membership=module, environment_membership=environment)
                proposals.append(OracleProposal("split_and_prune", node_id, None, None, candidate, train_evidence_id))
            elif float(plan.split_gates[node_id].detach()) == 1.0:
                proposals.append(OracleProposal(
                    "merge", node_id, None, None, plan.with_split(node_id, 0.0), train_evidence_id
                ))
        for source_type, membership, relevance, resolved in (
            ("module", plan.module_membership, module_relevance, resolved_module),
            ("environment", plan.environment_membership, environment_relevance, resolved_environment),
        ):
            for source in range(int(membership.shape[1])):
                if (
                    bool(membership[node_id, source] > 0)
                    and bool(resolved[node_id, source])
                    and bool(relevance[node_id, source] <= relevance_threshold)
                ):
                    edited = membership.clone()
                    edited[node_id, source] = 0.0
                    candidate = replace(
                        plan,
                        **{f"{source_type}_membership": edited},
                    )
                    proposals.append(OracleProposal(
                        "prune", node_id, source_type, source, candidate, train_evidence_id
                    ))
    return tuple(proposals)


def search_training_cover(
    full_access: AdaptiveCoverPlan,
    queries: torch.Tensor,
    *,
    module_present: torch.Tensor,
    environment_weights: torch.Tensor,
    observe: Callable[[AdaptiveCoverPlan], OracleObservation],
    propose: Callable[[AdaptiveCoverPlan], tuple[OracleProposal, ...]],
    limits: Mapping[str, RoleLimit],
    costs: MeasuredCostCoefficients,
    max_evaluations: int,
    minimum_cost_improvement: float = 0.01,
) -> OracleResult:
    """Greedy bounded search, accepting only protected adequate cheaper plans."""

    if max_evaluations < 1:
        raise ValueError("oracle evaluation cap must be positive")
    if not 0.0 <= minimum_cost_improvement < 1.0:
        raise ValueError("minimum cost improvement must be in [0,1)")
    if not limits:
        raise ValueError("at least one protected role gate is required")
    _qm, _qe, baseline_ledger = compile_cover_pairs(
        full_access, queries,
        module_present=module_present,
        environment_weights=environment_weights,
    )
    baseline = observe(full_access)
    if not baseline.reference_evidence_id or not baseline.teacher_checkpoint_id:
        raise ValueError("oracle baseline needs reference and teacher provenance IDs")
    allowed, reason = _adequate(baseline, baseline, limits)
    if not allowed:
        raise ValueError(f"full-access operator fails the oracle gate: {reason}")
    selected_plan = full_access
    selected_observation = baseline
    selected_ledger = baseline_ledger
    baseline_ms = costs.estimate_ms(full_access, baseline_ledger)
    selected_ms = baseline_ms
    history: list[OracleTrial] = []
    evaluated = 0
    while evaluated < max_evaluations:
        changed = False
        for proposal in propose(selected_plan):
            if evaluated >= max_evaluations:
                break
            evaluated += 1
            if proposal.train_evidence_id != baseline.reference_evidence_id:
                raise ValueError("oracle proposal must use the baseline training evidence")
            if proposal.plan.tree is not full_access.tree:
                raise ValueError("oracle cannot replace the declared candidate hierarchy")
            _qm, _qe, ledger = compile_cover_pairs(
                proposal.plan, queries,
                module_present=module_present,
                environment_weights=environment_weights,
            )
            observation = observe(proposal.plan)
            if observation.reference_evidence_id != baseline.reference_evidence_id:
                raise ValueError("oracle observation changed its reference evidence")
            if observation.teacher_checkpoint_id != baseline.teacher_checkpoint_id:
                raise ValueError("oracle observation changed its frozen teacher")
            adequate, reason = _adequate(observation, baseline, limits)
            estimated_ms = costs.estimate_ms(proposal.plan, ledger)
            accepted = adequate and estimated_ms < selected_ms * (1.0 - minimum_cost_improvement)
            if adequate and not accepted:
                reason = "no_measured_cost_gain"
            history.append(OracleTrial(proposal, observation, ledger, estimated_ms, accepted, reason))
            if accepted:
                selected_plan = proposal.plan
                selected_observation = observation
                selected_ledger = ledger
                selected_ms = estimated_ms
                changed = True
                break
        if not changed:
            break
    return OracleResult(
        baseline, baseline_ledger, baseline_ms,
        selected_plan, selected_observation, selected_ledger, selected_ms,
        tuple(history),
    )


__all__ = [
    "MeasuredCostCoefficients", "OracleObservation", "OracleProposal", "OracleResult",
    "OracleTrial", "RoleError", "RoleLimit", "local_cover_proposals", "search_training_cover",
]
