"""Bounded train-only search for an interaction cover.

The caller supplies frozen-model observations on training probes. Reference
sufficiency and teacher preservation have separate gates; this module never
reads labels itself and never runs at deployment. Its row counts are exact for
the declared probe queries; complete cost uses measured coefficients and is
only a proposal estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import torch

from .adaptive_interaction_cover import (
    AdaptiveCoverPlan,
    CoverPairLedger,
    MechanismPlan,
    compile_cover_pairs,
)


@dataclass(frozen=True)
class RoleError:
    """One protected role's optional reference and frozen-teacher errors."""

    reference: float | None
    teacher: float | None
    resolved: bool


@dataclass(frozen=True)
class OracleObservation:
    roles: Mapping[str, RoleError]
    reference_evidence_id: str | None
    teacher_checkpoint_id: str
    training_evidence_id: str | None = None


@dataclass(frozen=True)
class RoleLimit:
    absolute_error: float
    allowed_increase: float


@dataclass(frozen=True)
class TeacherDistortionLimit:
    """Maximum normalized candidate-to-teacher distortion for one role."""

    maximum_distortion: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_distortion) or self.maximum_distortion < 0.0:
            raise ValueError("teacher distortion limits must be finite and nonnegative")


@dataclass(frozen=True)
class CostWorkAccounting:
    """Separate organizer scoring work, plan structure, and executed rows."""

    candidate_nodes: int
    scored_candidate_source_rows: int
    active_frontier_nodes: int
    nonredundant_packets: int
    unique_source_receiver_pairs: int
    dense_masked_executed_rows: int
    subset_executed_rows: int
    packed_executed_rows: int
    dense_masked_padded_rows: int

    def as_dict(self) -> dict[str, int]:
        return {
            "candidate_nodes": self.candidate_nodes,
            "scored_candidate_source_rows": self.scored_candidate_source_rows,
            "active_frontier_nodes": self.active_frontier_nodes,
            "nonredundant_packets": self.nonredundant_packets,
            "unique_source_receiver_pairs": self.unique_source_receiver_pairs,
            "dense_masked_executed_rows": self.dense_masked_executed_rows,
            "subset_executed_rows": self.subset_executed_rows,
            "packed_executed_rows": self.packed_executed_rows,
            "dense_masked_padded_rows": self.dense_masked_padded_rows,
        }


@dataclass(frozen=True)
class MeasuredCostCoefficients:
    """Measured v1 compatibility or v2 candidate-work execution coefficients.

    Positional v1 records preserve their historical active-group formula.
    Typed structural estimates must explicitly opt into ``candidate_work_v2``
    and provide the all-candidate scoring coefficients and executor fixed
    costs measured for that path.
    """

    measurement_id: str
    preparation_ms: float
    organization_ms_per_node: float  # legacy v1 only: per active source-bearing node
    route_ms_per_path: float
    qm_ms_per_row: float
    qe_ms_per_row: float
    pack_ms_per_row: float
    wrapper_ms: float
    cost_model: Literal["legacy_active_group_v1", "candidate_work_v2"] = "legacy_active_group_v1"
    organization_ms_per_candidate_node: float | None = None
    organization_ms_per_candidate_source_row: float | None = None
    packet_ms_per_packet: float = 0.0
    dense_dispatch_ms: float = 0.0
    dense_launch_ms: float = 0.0
    subset_dispatch_ms: float = 0.0
    subset_launch_ms: float = 0.0
    packed_dispatch_ms: float = 0.0
    packed_launch_ms: float = 0.0
    subset_gather_ms_per_row: float = 0.0

    def __post_init__(self) -> None:
        if not self.measurement_id:
            raise ValueError("cost coefficients require a measured provenance ID")
        for key in (
            "preparation_ms", "organization_ms_per_node", "route_ms_per_path",
            "qm_ms_per_row", "qe_ms_per_row", "pack_ms_per_row", "wrapper_ms",
            "packet_ms_per_packet", "dense_dispatch_ms", "dense_launch_ms",
            "subset_dispatch_ms", "subset_launch_ms", "packed_dispatch_ms",
            "packed_launch_ms", "subset_gather_ms_per_row",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0.0:
                raise ValueError(f"cost coefficient {key} must be finite and nonnegative")

        for key in (
            "organization_ms_per_candidate_node",
            "organization_ms_per_candidate_source_row",
        ):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"cost coefficient {key} must be finite and nonnegative")
        if self.cost_model not in {"legacy_active_group_v1", "candidate_work_v2"}:
            raise ValueError("unknown measured cost model")
        if self.cost_model == "candidate_work_v2" and (
            self.organization_ms_per_candidate_node is None
            or self.organization_ms_per_candidate_source_row is None
        ):
            raise ValueError(
                "candidate_work_v2 requires measured node and scored-source-row coefficients"
            )

    @staticmethod
    def work_accounting(
        plan: AdaptiveCoverPlan | MechanismPlan,
        ledger: CoverPairLedger,
        *,
        phase: str | None = None,
    ) -> CostWorkAccounting:
        candidate_nodes = len(plan.tree.nodes)
        module_sources = (
            int(plan.module_membership.shape[1])
            if isinstance(plan, AdaptiveCoverPlan)
            else int(plan.module_present.numel())
        )
        environment_sources = (
            int(plan.environment_membership.shape[1])
            if isinstance(plan, AdaptiveCoverPlan)
            else int(plan.environment_count)
        )
        scored_candidate_source_rows = candidate_nodes * (
            3 * module_sources + 2 * environment_sources
        )
        if isinstance(plan, MechanismPlan):
            summaries = [
                plan.frontier_summary(mechanism, phase)
                for mechanism in ("MM", "ME", "EM", "QM", "QE")
            ]
            active_frontier_nodes = sum(item.source_bearing_active_nodes for item in summaries)
            nonredundant_packets = sum(item.nonredundant_packet_count for item in summaries)
        else:
            active_frontier_nodes = plan.active_group_count()
            active = (
                plan.tree.access(plan.tree.universe.coordinates, plan.split_gates) > 0
            ).any(dim=0)
            signatures = {
                (
                    tuple(float(value) for value in plan.module_membership[index].detach().cpu().tolist()),
                    tuple(float(value) for value in plan.environment_membership[index].detach().cpu().tolist()),
                )
                for index in torch.nonzero(active, as_tuple=False).flatten().tolist()
            }
            nonredundant_packets = len(signatures)
        unique_pairs = int(ledger.qm_unique_rows + ledger.qe_unique_rows)
        dense_rows = int(ledger.qm_rectangular_rows + ledger.qe_rectangular_rows)
        return CostWorkAccounting(
            candidate_nodes,
            scored_candidate_source_rows,
            int(active_frontier_nodes),
            int(nonredundant_packets),
            unique_pairs,
            dense_rows,
            unique_pairs,
            unique_pairs,
            max(0, dense_rows - unique_pairs),
        )

    def estimate_ms(
        self,
        plan: AdaptiveCoverPlan | MechanismPlan,
        ledger: CoverPairLedger,
        *,
        executor: Literal["dense_masked", "rectangular_subset", "packed", "full_access"] = "packed",
        phase: str | None = None,
    ) -> float:
        if self.cost_model == "legacy_active_group_v1":
            if isinstance(plan, MechanismPlan):
                raise ValueError(
                    "legacy active-group coefficients cannot estimate typed plans; fit candidate_work_v2"
                )
            return (
                self.preparation_ms
                + self.organization_ms_per_node * plan.active_group_count()
                + self.route_ms_per_path * (ledger.qm_raw_paths + ledger.qe_raw_paths)
                + self.qm_ms_per_row * ledger.qm_unique_rows
                + self.qe_ms_per_row * ledger.qe_unique_rows
                + self.pack_ms_per_row * (ledger.qm_unique_rows + ledger.qe_unique_rows)
                + self.wrapper_ms
            )
        if executor not in {"dense_masked", "rectangular_subset", "packed", "full_access"}:
            raise ValueError("unsupported executor for candidate-work cost estimate")
        work = self.work_accounting(plan, ledger, phase=phase)
        if executor == "full_access":
            executor = "dense_masked"
        if executor == "dense_masked":
            dispatch_ms = self.dense_dispatch_ms
            launch_ms = self.dense_launch_ms
            qm_rows = int(ledger.qm_rectangular_rows)
            qe_rows = int(ledger.qe_rectangular_rows)
            gather_ms = 0.0
            pack_ms = 0.0
        elif executor == "rectangular_subset":
            dispatch_ms = self.subset_dispatch_ms
            launch_ms = self.subset_launch_ms
            qm_rows = int(ledger.qm_unique_rows)
            qe_rows = int(ledger.qe_unique_rows)
            gather_ms = self.subset_gather_ms_per_row * (qm_rows + qe_rows)
            pack_ms = 0.0
        else:
            dispatch_ms = self.packed_dispatch_ms
            launch_ms = self.packed_launch_ms
            qm_rows = int(ledger.qm_unique_rows)
            qe_rows = int(ledger.qe_unique_rows)
            gather_ms = 0.0
            pack_ms = self.pack_ms_per_row * (qm_rows + qe_rows)
        return (
            self.preparation_ms
            + float(self.organization_ms_per_candidate_node) * work.candidate_nodes
            + float(self.organization_ms_per_candidate_source_row) * work.scored_candidate_source_rows
            + self.packet_ms_per_packet * work.nonredundant_packets
            + self.route_ms_per_path * (ledger.qm_raw_paths + ledger.qe_raw_paths)
            + self.qm_ms_per_row * qm_rows
            + self.qe_ms_per_row * qe_rows
            + gather_ms + pack_ms + dispatch_ms + launch_ms + self.wrapper_ms
        )

    def estimate_breakdown(
        self,
        plan: AdaptiveCoverPlan | MechanismPlan,
        ledger: CoverPairLedger,
        *,
        executor: Literal["dense_masked", "rectangular_subset", "packed", "full_access"] = "packed",
        phase: str | None = None,
    ) -> dict[str, int | float | str]:
        """Expose scorer work and executor row alternatives with estimated time."""

        work = self.work_accounting(plan, ledger, phase=phase)
        return {
            "measurement_id": self.measurement_id,
            "cost_model": self.cost_model,
            "executor": executor,
            **work.as_dict(),
            "estimated_ms": self.estimate_ms(
                plan, ledger, executor=executor, phase=phase
            ),
        }


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
    gate_passed: bool
    work: CostWorkAccounting | None = None


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
    evidence_mode: str = "reference_sufficiency"
    baseline_work: CostWorkAccounting | None = None
    selected_work: CostWorkAccounting | None = None


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
        if not math.isfinite(current.reference) or current.teacher is None or not math.isfinite(current.teacher):
            return False, f"nonfinite_role:{role}"
        if current.reference > limit.absolute_error:
            return False, f"absolute_error:{role}"
        if current.reference > reference.reference + limit.allowed_increase:
            return False, f"relative_to_full:{role}"
    return True, "adequate"


def teacher_preservation_gate(
    candidate: OracleObservation,
    limits: Mapping[str, TeacherDistortionLimit],
) -> tuple[bool, str]:
    """Check candidate-to-teacher distortion without consulting references.

    In this mode ``RoleError.teacher`` is the measured normalized discrepancy
    between the candidate cover and the same frozen full-access teacher.  The
    reference field and ``resolved`` flag are intentionally irrelevant.
    """

    for role, limit in limits.items():
        current = candidate.roles.get(role)
        if current is None:
            return False, f"missing_teacher_role:{role}"
        if current.teacher is None or not math.isfinite(current.teacher):
            return False, f"nonfinite_teacher_role:{role}"
        if current.teacher > limit.maximum_distortion:
            return False, f"teacher_distortion:{role}"
    return True, "teacher_preserved"


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
    limits: Mapping[str, RoleLimit] | None = None,
    costs: MeasuredCostCoefficients,
    max_evaluations: int,
    minimum_cost_improvement: float = 0.01,
    evidence_mode: Literal["reference_sufficiency", "teacher_preservation"] = "reference_sufficiency",
    teacher_limits: Mapping[str, TeacherDistortionLimit] | None = None,
    explore_alternatives: bool = False,
) -> OracleResult:
    """Greedy bounded search under an explicitly selected evidence contract.

    ``reference_sufficiency`` preserves the historical physical-reference
    gate. ``teacher_preservation`` accepts only covers within explicit,
    normalized distortion limits from the frozen full-access teacher; it does
    not require resolved physical reference roles. Proposal cost remains an
    estimate in both modes.
    """

    if max_evaluations < 1:
        raise ValueError("oracle evaluation cap must be positive")
    if not 0.0 <= minimum_cost_improvement < 1.0:
        raise ValueError("minimum cost improvement must be in [0,1)")
    if evidence_mode not in {"reference_sufficiency", "teacher_preservation"}:
        raise ValueError("unknown cover-oracle evidence mode")
    if evidence_mode == "reference_sufficiency":
        if not limits or teacher_limits is not None:
            raise ValueError("reference sufficiency requires reference limits only")
    elif not teacher_limits or limits not in (None, {}):
        raise ValueError("teacher preservation requires teacher distortion limits only")
    _qm, _qe, baseline_ledger = compile_cover_pairs(
        full_access, queries,
        module_present=module_present,
        environment_weights=environment_weights,
    )
    baseline = observe(full_access)
    if not baseline.teacher_checkpoint_id:
        raise ValueError("oracle baseline needs a frozen teacher checkpoint ID")
    if evidence_mode == "reference_sufficiency":
        if not baseline.reference_evidence_id:
            raise ValueError("reference-sufficiency baseline needs reference provenance")
        evidence_id = baseline.reference_evidence_id
        allowed, reason = _adequate(baseline, baseline, limits or {})
    else:
        if not baseline.training_evidence_id:
            raise ValueError("teacher-preservation baseline needs training evidence provenance")
        evidence_id = baseline.training_evidence_id
        allowed, reason = teacher_preservation_gate(baseline, teacher_limits or {})
    if not allowed:
        raise ValueError(f"full-access operator fails the {evidence_mode} gate: {reason}")
    selected_plan = full_access
    selected_observation = baseline
    selected_ledger = baseline_ledger
    baseline_ms = costs.estimate_ms(full_access, baseline_ledger)
    baseline_work = costs.work_accounting(full_access, baseline_ledger)
    selected_ms = baseline_ms
    selected_work = baseline_work
    history: list[OracleTrial] = []
    evaluated = 0
    while evaluated < max_evaluations:
        changed = False
        best_in_round: tuple[AdaptiveCoverPlan, OracleObservation, CoverPairLedger, float] | None = None
        for proposal in propose(selected_plan):
            if evaluated >= max_evaluations:
                break
            evaluated += 1
            if proposal.train_evidence_id != evidence_id:
                raise ValueError("oracle proposal must use the baseline training evidence")
            if proposal.plan.tree is not full_access.tree:
                raise ValueError("oracle cannot replace the declared candidate hierarchy")
            _qm, _qe, ledger = compile_cover_pairs(
                proposal.plan, queries,
                module_present=module_present,
                environment_weights=environment_weights,
            )
            observation = observe(proposal.plan)
            observed_evidence_id = (
                observation.reference_evidence_id
                if evidence_mode == "reference_sufficiency"
                else observation.training_evidence_id
            )
            if observed_evidence_id != evidence_id:
                evidence_name = "reference" if evidence_mode == "reference_sufficiency" else "training"
                raise ValueError(f"oracle observation changed its {evidence_name} evidence")
            if observation.teacher_checkpoint_id != baseline.teacher_checkpoint_id:
                raise ValueError("oracle observation changed its frozen teacher")
            if evidence_mode == "reference_sufficiency":
                gate_passed, reason = _adequate(observation, baseline, limits or {})
            else:
                gate_passed, reason = teacher_preservation_gate(observation, teacher_limits or {})
            estimated_ms = costs.estimate_ms(proposal.plan, ledger)
            work = costs.work_accounting(proposal.plan, ledger)
            accepted = gate_passed and estimated_ms < selected_ms * (1.0 - minimum_cost_improvement)
            if gate_passed and not accepted:
                reason = "no_estimated_cost_gain"
            history.append(OracleTrial(
                proposal, observation, ledger, estimated_ms, accepted, reason, gate_passed, work
            ))
            if accepted and (best_in_round is None or estimated_ms < best_in_round[3]):
                best_in_round = (proposal.plan, observation, ledger, estimated_ms)
                if not explore_alternatives:
                    break
        if best_in_round is not None:
            selected_plan, selected_observation, selected_ledger, selected_ms = best_in_round
            selected_work = costs.work_accounting(selected_plan, selected_ledger)
            changed = True
        if not changed:
            break
    return OracleResult(
        baseline, baseline_ledger, baseline_ms,
        selected_plan, selected_observation, selected_ledger, selected_ms,
        tuple(history), evidence_mode, baseline_work, selected_work,
    )


__all__ = [
    "CostWorkAccounting", "MeasuredCostCoefficients", "OracleObservation", "OracleProposal", "OracleResult",
    "OracleTrial", "RoleError", "RoleLimit", "TeacherDistortionLimit",
    "local_cover_proposals", "search_training_cover", "teacher_preservation_gate",
]
