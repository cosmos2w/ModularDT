from __future__ import annotations

import torch

from honf_forward_core.interface_fields.adaptive_cover_oracle import (
    MeasuredCostCoefficients,
    OracleObservation,
    OracleProposal,
    RoleError,
    TeacherDistortionLimit,
    search_training_cover,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    AdaptiveCoverPlan,
    CaseLocalReceiverTree,
    ReceiverAnchorUniverse,
)


def _fixture() -> AdaptiveCoverPlan:
    device = torch.device("cpu")
    universe = ReceiverAnchorUniverse(
        coordinates=torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]], device=device
        ),
        weights=torch.ones(4, device=device),
        roles=torch.zeros(4, dtype=torch.long, device=device),
        coordinate_scale=torch.ones(2, device=device),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=3, min_leaf_anchors=1)
    return AdaptiveCoverPlan.full_access(
        tree,
        module_present=torch.tensor([1.0, 1.0], device=device),
        environment_count=3,
    )


def _costs() -> MeasuredCostCoefficients:
    return MeasuredCostCoefficients("measured-fixture", 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)


def test_teacher_preservation_accepts_unresolved_reference_roles_and_keeps_alternatives() -> None:
    plan = _fixture()
    evidence_id = "windfarm-training-probe-fixture"
    teacher_id = "Run2103-e2475-fixture"

    def observe(_candidate: AdaptiveCoverPlan) -> OracleObservation:
        return OracleObservation(
            roles={"wake": RoleError(reference=None, teacher=0.02, resolved=False)},
            reference_evidence_id=None,
            teacher_checkpoint_id=teacher_id,
            training_evidence_id=evidence_id,
        )

    proposals = tuple(
        OracleProposal("exploratory", index, None, None, plan, evidence_id)
        for index in range(2)
    )
    result = search_training_cover(
        plan,
        plan.tree.universe.coordinates,
        module_present=torch.tensor([1.0, 1.0]),
        environment_weights=torch.ones(3),
        observe=observe,
        propose=lambda _current: proposals,
        costs=_costs(),
        max_evaluations=2,
        evidence_mode="teacher_preservation",
        teacher_limits={"wake": TeacherDistortionLimit(0.05)},
        explore_alternatives=True,
    )

    assert result.evidence_mode == "teacher_preservation"
    assert result.selected_plan is plan
    assert len(result.trials) == 2
    assert all(trial.gate_passed for trial in result.trials)
    assert all(trial.reason == "no_estimated_cost_gain" for trial in result.trials)
    assert all(trial.observation.roles["wake"].reference is None for trial in result.trials)


def test_teacher_preservation_requires_a_finite_distortion_within_declared_limit() -> None:
    plan = _fixture()
    evidence_id = "windfarm-training-probe-fixture"
    teacher_id = "Run2103-e2475-fixture"
    observations = 0

    def observe(_candidate: AdaptiveCoverPlan) -> OracleObservation:
        nonlocal observations
        observations += 1
        value = 0.0 if observations == 1 else 0.06
        return OracleObservation(
            roles={"rotor": RoleError(reference=None, teacher=value, resolved=False)},
            reference_evidence_id=None,
            teacher_checkpoint_id=teacher_id,
            training_evidence_id=evidence_id,
        )

    result = search_training_cover(
        plan,
        plan.tree.universe.coordinates,
        module_present=torch.tensor([1.0, 1.0]),
        environment_weights=torch.ones(3),
        observe=observe,
        propose=lambda current: (
            OracleProposal("candidate", 0, None, None, current, evidence_id),
        ),
        costs=_costs(),
        max_evaluations=1,
        evidence_mode="teacher_preservation",
        teacher_limits={"rotor": TeacherDistortionLimit(0.05)},
    )

    assert result.selected_plan is plan
    assert len(result.trials) == 1
    assert not result.trials[0].gate_passed
    assert not result.trials[0].accepted
    assert result.trials[0].reason == "teacher_distortion:rotor"


def test_teacher_preservation_rejects_changed_training_evidence() -> None:
    plan = _fixture()
    observations = 0

    def observe(_candidate: AdaptiveCoverPlan) -> OracleObservation:
        nonlocal observations
        observations += 1
        return OracleObservation(
            roles={"wake": RoleError(reference=None, teacher=0.0, resolved=False)},
            reference_evidence_id=None,
            teacher_checkpoint_id="Run2103-e2475-fixture",
            training_evidence_id="train-a" if observations == 1 else "train-b",
        )

    try:
        search_training_cover(
            plan,
            plan.tree.universe.coordinates,
            module_present=torch.tensor([1.0, 1.0]),
            environment_weights=torch.ones(3),
            observe=observe,
            propose=lambda current: (
                OracleProposal("candidate", 0, None, None, current, "train-a"),
            ),
            costs=_costs(),
            max_evaluations=1,
            evidence_mode="teacher_preservation",
            teacher_limits={"wake": TeacherDistortionLimit(0.05)},
        )
    except ValueError as error:
        assert "changed its training evidence" in str(error)
    else:
        raise AssertionError("oracle must reject a proposal observation from another training panel")
