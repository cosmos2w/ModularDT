"""Exact case-local cover endpoints, duplicate accounting, and invariance."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.adaptive_cover_oracle import (
    MeasuredCostCoefficients,
    OracleObservation,
    OracleProposal,
    RoleError,
    RoleLimit,
    local_cover_proposals,
    search_training_cover,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    AdaptiveCoverPlan,
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
    compile_cover_pairs,
    compile_mechanism_execution_view,
)


def _fixture(device: torch.device) -> tuple[CaseLocalReceiverTree, AdaptiveCoverPlan]:
    coordinates = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], device=device)
    universe = ReceiverAnchorUniverse(
        coordinates=coordinates,
        weights=torch.ones(4, device=device),
        roles=torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device),
        coordinate_scale=torch.tensor([10.0, 4.0], device=device),
    )
    tree = CaseLocalReceiverTree.build(
        universe, max_nodes=7, min_leaf_anchors=1, overlap_fraction=0.1
    )
    plan = AdaptiveCoverPlan.full_access(tree, torch.tensor([1.0, 1.0, 0.0], device=device), 2)
    assert tree.nodes[0].left == 1 and tree.nodes[0].right == 2
    module = plan.module_membership.clone()
    environment = plan.environment_membership.clone()
    module[1] = torch.tensor([1.0, 0.0, 0.0], device=device)
    module[2] = torch.tensor([0.0, 1.0, 0.0], device=device)
    environment[1] = torch.tensor([1.0, 0.0], device=device)
    environment[2] = torch.tensor([0.0, 1.0], device=device)
    return tree, replace(plan, module_membership=module, environment_membership=environment)


def test_case_cover_endpoints_transition_count_and_unique_physical_pairs() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tree, initial = _fixture(device)
    queries = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], device=device)
    module_present = torch.tensor([1.0, 1.0, 0.0], device=device)
    environment_weights = torch.tensor([2.0, 3.0], device=device)

    full = initial.access(queries)
    assert full.active_groups_on_anchors == 1
    assert full.transition_nodes == 0
    assert torch.equal(full.module_source[:, 2], torch.zeros(2, device=device))
    _, _, full_ledger = compile_cover_pairs(
        initial, queries, module_present=module_present, environment_weights=environment_weights
    )
    assert (full_ledger.qm_unique_rows, full_ledger.qe_unique_rows) == (4, 4)
    assert (full_ledger.query_degree_sum, full_ledger.query_degree_max, full_ledger.query_count) == (2, 1, 2)

    split = initial.with_split(0, 1.0)
    split_access = split.access(queries)
    assert split_access.active_groups_on_anchors == 2
    assert split_access.transition_nodes == 0
    assert torch.equal(split_access.module_source, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device))
    _, _, split_ledger = compile_cover_pairs(
        split, queries, module_present=module_present, environment_weights=environment_weights
    )
    assert (split_ledger.qm_unique_rows, split_ledger.qe_unique_rows) == (2, 2)
    assert (split_ledger.query_degree_sum, split_ledger.query_degree_max) == (2, 1)
    assert split_ledger.qm_rectangular_rows == 6  # Dense evaluates one padded module slot.

    transition = initial.with_split(0, 0.5)
    transition_access = transition.access(queries)
    assert transition_access.active_groups_on_anchors == 3
    assert transition_access.transition_nodes == 1
    assert torch.allclose(transition_access.receiver_group.sum(dim=1), torch.ones(2, device=device))
    _, _, transition_ledger = compile_cover_pairs(
        transition, queries, module_present=module_present, environment_weights=environment_weights
    )
    assert transition_ledger.qm_raw_paths > transition_ledger.qm_unique_rows
    assert transition_ledger.qm_unique_rows == 4
    assert transition_ledger.qe_unique_rows == 4
    assert (transition_ledger.query_degree_sum, transition_ledger.query_degree_max) == (4, 2)
    assert len(tree.nodes) <= 7


def test_typed_plan_hash_serialization_and_missing_key_full_access_ledger() -> None:
    tree, legacy = _fixture(torch.device("cpu"))
    module_present = torch.tensor([1.0, 1.0, 0.0])
    plan = MechanismPlan.from_legacy(legacy, module_present)
    reversed_plan = MechanismPlan(
        plan.tree,
        plan.split_gates,
        plan.module_present,
        plan.environment_count,
        dict(reversed(tuple(plan.permissions.items()))),
    )
    assert plan.canonical_hash() == reversed_plan.canonical_hash()
    restored = MechanismPlan.from_dict(plan.to_dict())
    assert restored.canonical_hash() == plan.canonical_hash()
    assert restored.explicit_bypass_keys == plan.explicit_bypass_keys == ()
    for mechanism in ("MM", "ME", "EM", "QM", "QE"):
        torch.testing.assert_close(
            restored.permission_matrix(mechanism), plan.permission_matrix(mechanism)
        )

    implicit = MechanismPlan(tree, torch.zeros(len(tree.nodes)), module_present, 2, {})
    assert set(implicit.explicit_bypass_keys) == {"MM", "ME", "EM", "QM", "QE"}
    assert implicit.permission_status("QM") == "full_access_bypass_missing_key"
    torch.testing.assert_close(
        implicit.access_for("QM", tree.universe.coordinates),
        torch.ones((tree.universe.coordinates.shape[0], 3)) * module_present[None],
    )
    payload = implicit.to_dict()
    assert set(payload["explicit_bypass_keys"]) == {"MM", "ME", "EM", "QM", "QE"}
    assert MechanismPlan.from_dict(payload).canonical_hash() == implicit.canonical_hash()


def test_compiled_execution_view_keeps_exact_plan_metadata_runtime_only(monkeypatch) -> None:
    assert torch.cuda.is_available(), "this execution regression is assigned to physical GPU 2"
    device = torch.device("cuda:0")
    tree, _legacy = _fixture(device)
    module_present = torch.tensor([1.0, 1.0, 0.0], device=device)
    plan = MechanismPlan.full_access(tree, module_present, 2, phase="P0").with_split(0, 1.0)
    queries = torch.tensor([[-0.1, 0.0], [0.0, 0.0], [0.1, 0.0]], device=device)
    reference_access = tree.access(queries, plan.split_gates)
    calls = {"tree_access": 0}
    original_access = CaseLocalReceiverTree.access

    def counted_access(self, queries, split_gates):
        calls["tree_access"] += 1
        return original_access(self, queries, split_gates)

    monkeypatch.setattr(CaseLocalReceiverTree, "access", counted_access)
    view = compile_mechanism_execution_view(
        plan,
        module_present=module_present,
        environment_count=2,
        phase="P0",
    )

    assert calls["tree_access"] == 1
    assert view.matches(plan, phase="P0")
    assert not view.matches(plan, phase="P1")
    assert view.active_node_indices == (1, 2)
    assert view.hard_execution_node_mask == (True, True, True, False, False, False, False)
    assert view.parent_child_relations == ((0, 1, 2), (1, 3, 4), (2, 5, 6))
    assert view.is_full_access("QM") and view.is_full_access("QE")
    assert not view.has_differentiable_routing(("QM", "QE"))
    assert view.source_indices("QM") == (
        (), (0, 1), (0, 1), (), (), (), ()
    )
    assert view.action_classes("QM") == ((1, 2),)
    torch.testing.assert_close(
        view.receiver_access(queries, plan.split_gates), reference_access, rtol=0.0, atol=0.0
    )
    qm = view.summary("QM")
    assert qm.raw_active_frontier_nodes == 2
    assert qm.source_bearing_active_nodes == 2
    assert qm.nonredundant_packet_count == 1
    assert qm.source_union_count == 2
    # Compiled views are execution state only and do not alter the serialized
    # plan contract or its canonical identity.
    restored = MechanismPlan.from_dict(plan.to_dict())
    assert restored.canonical_hash() == plan.canonical_hash()
    with torch.no_grad():
        plan.split_gates[0].zero_()
    assert not view.matches(plan, phase="P0")


@pytest.mark.parametrize("root_gate", [0.0, 0.45, 1.0])
@pytest.mark.parametrize("trainable_gate", [False, True])
def test_compiled_receiver_access_matches_hard_and_soft_geometry_gradients(
    root_gate: float, trainable_gate: bool,
) -> None:
    """Prepared split levels preserve exact values and live geometry gradients."""

    assert torch.cuda.is_available(), "this execution regression is assigned to physical GPU 2"
    device = torch.device("cuda:0")
    initial_tree, _legacy = _fixture(device)
    coordinates = initial_tree.universe.coordinates.detach().clone().requires_grad_(True)
    coordinate_scale = initial_tree.universe.coordinate_scale.detach().clone().requires_grad_(True)
    universe = ReceiverAnchorUniverse(
        coordinates,
        initial_tree.universe.weights,
        initial_tree.universe.roles,
        coordinate_scale,
    )
    tree = CaseLocalReceiverTree(
        universe,
        initial_tree.nodes,
        initial_tree.overlap_fraction,
        initial_tree.capacity_saturated,
    )
    module_present = torch.tensor([1.0, 1.0, 0.0], device=device)
    plan = MechanismPlan.full_access(tree, module_present, 2, phase="P0")
    gate_values = torch.zeros(len(tree.nodes), device=device)
    gate_values[0] = root_gate
    gate_values[1] = 0.35
    gate_values[2] = 0.65
    if trainable_gate:
        gate_values.requires_grad_()
    plan = replace(plan, split_gates=gate_values)
    view = compile_mechanism_execution_view(
        plan,
        module_present=module_present,
        environment_count=2,
        phase="P0",
    )
    queries = torch.tensor(
        [[-1.7, -0.2], [-0.4, 0.1], [0.3, 0.0], [1.8, 0.3]],
        device=device,
        requires_grad=True,
    )
    expected = tree.access(queries, plan.split_gates)
    actual = view.receiver_access(queries, plan.split_gates)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    weights = torch.arange(1, actual.numel() + 1, device=device, dtype=actual.dtype).reshape_as(actual)
    inputs = [queries, coordinates, coordinate_scale]
    if trainable_gate:
        inputs.append(plan.split_gates)
    expected_loss = (expected * weights).sum()
    actual_loss = (actual * weights).sum()
    expected_grads = (
        torch.autograd.grad(expected_loss, inputs, retain_graph=True, allow_unused=True)
        if expected_loss.requires_grad else (None,) * len(inputs)
    )
    actual_grads = (
        torch.autograd.grad(actual_loss, inputs, retain_graph=True, allow_unused=True)
        if actual_loss.requires_grad else (None,) * len(inputs)
    )
    for expected_grad, actual_grad, input_tensor in zip(expected_grads, actual_grads, inputs, strict=True):
        if expected_grad is None:
            expected_grad = torch.zeros_like(input_tensor)
        if actual_grad is None:
            actual_grad = torch.zeros_like(input_tensor)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2.0e-6, atol=2.0e-7)


def test_cover_continuity_chunk_order_source_relabel_and_empty_fallback() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tree, plan = _fixture(device)
    query = torch.tensor([[-0.1, 0.0], [0.0, 0.0], [0.1, 0.0]], device=device)
    lower = plan.with_split(0, 0.0).access(query).module_source
    almost_lower = plan.with_split(0, 1.0e-4).access(query).module_source
    upper = plan.with_split(0, 1.0).access(query).module_source
    almost_upper = plan.with_split(0, 1.0 - 1.0e-4).access(query).module_source
    assert float((almost_lower - lower).abs().max()) < 1.0e-6
    assert float((almost_upper - upper).abs().max()) < 1.0e-6

    split = plan.with_split(0, 1.0)
    whole = split.access(query)
    reordered = split.access(query[[2, 0, 1]])
    torch.testing.assert_close(whole.module_source[[2, 0, 1]], reordered.module_source)
    assert whole.active_groups_on_anchors == reordered.active_groups_on_anchors
    chunks = torch.cat((split.access(query[:1]).environment_source, split.access(query[1:]).environment_source))
    torch.testing.assert_close(whole.environment_source, chunks)

    permuted = replace(split, module_membership=split.module_membership[:, [1, 0, 2]])
    torch.testing.assert_close(
        split.access(query).module_source[:, [1, 0, 2]], permuted.access(query).module_source
    )
    assert split.access(query).active_groups_on_anchors == permuted.access(query).active_groups_on_anchors

    empty_environment = replace(split, environment_membership=torch.zeros_like(split.environment_membership))
    _, packed_environment, ledger = compile_cover_pairs(
        empty_environment,
        query,
        module_present=torch.tensor([1.0, 1.0, 0.0], device=device),
        environment_weights=torch.tensor([2.0, 3.0], device=device),
    )
    assert ledger.environment_fallback_queries == 3
    assert packed_environment.unique_pair_count == 6
    assert torch.all(packed_environment.prior > 0)
    almost_empty = replace(
        split,
        environment_membership=torch.full_like(split.environment_membership, 1.0e-8),
    )
    _, near_environment, near_ledger = compile_cover_pairs(
        almost_empty,
        query,
        module_present=torch.tensor([1.0, 1.0, 0.0], device=device),
        environment_weights=torch.tensor([2.0, 3.0], device=device),
    )
    assert near_ledger.environment_fallback_queries == 3
    torch.testing.assert_close(
        near_environment.prior / near_environment.prior.sum(),
        packed_environment.prior / packed_environment.prior.sum(),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_receiver_tree_capacity_and_invalid_measure_are_explicit() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tree, plan = _fixture(device)
    small = CaseLocalReceiverTree.build(tree.universe, max_nodes=3, min_leaf_anchors=1)
    assert small.capacity_saturated and len(small.nodes) == 3
    with pytest.raises(ValueError, match="positive"):
        ReceiverAnchorUniverse(
            tree.universe.coordinates,
            torch.tensor([1.0, 0.0, 1.0, 1.0], device=device),
            tree.universe.roles,
            tree.universe.coordinate_scale,
        )
    with pytest.raises(ValueError, match="nonempty"):
        compile_cover_pairs(
            plan,
            tree.universe.coordinates,
            module_present=torch.ones(3, device=device),
            environment_weights=torch.empty(0, device=device),
        )


def test_cost_model_charges_active_cover_resolution_not_fixed_tree_capacity() -> None:
    """A split with no row benefit still adds measured per-group work."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tree, full = _fixture(device)
    split = full.with_split(0, 1.0)
    queries = tree.universe.coordinates
    module_present = torch.tensor([1.0, 1.0, 0.0], device=device)
    environment_weights = torch.tensor([2.0, 3.0], device=device)
    _, _, full_ledger = compile_cover_pairs(
        full, queries, module_present=module_present, environment_weights=environment_weights
    )
    _, _, split_ledger = compile_cover_pairs(
        split, queries, module_present=module_present, environment_weights=environment_weights
    )
    cost = MeasuredCostCoefficients("synthetic-cost-fixture", 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert len(full.tree.nodes) == len(split.tree.nodes)
    assert full.active_group_count() == 1
    assert split.active_group_count() == 2
    assert cost.estimate_ms(split, split_ledger) - cost.estimate_ms(full, full_ledger) == 2.0


def test_candidate_work_v2_records_full_scoring_work_and_executor_fixed_costs() -> None:
    device = torch.device("cpu")
    tree, legacy_plan = _fixture(device)
    plan = MechanismPlan.from_legacy(
        legacy_plan, torch.tensor([1.0, 1.0, 0.0], device=device)
    )
    queries = tree.universe.coordinates
    module_present = torch.tensor([1.0, 1.0, 0.0], device=device)
    environment_weights = torch.tensor([2.0, 3.0], device=device)
    _qm, _qe, ledger = compile_cover_pairs(
        plan,
        queries,
        module_present=module_present,
        environment_weights=environment_weights,
    )
    cost = MeasuredCostCoefficients(
        "synthetic-candidate-work-fixture",
        0.0,
        99.0,  # ignored by v2; retained solely for v1 record compatibility
        0.0,
        1.0,
        1.0,
        0.5,
        0.0,
        cost_model="candidate_work_v2",
        organization_ms_per_candidate_node=0.2,
        organization_ms_per_candidate_source_row=0.01,
        packet_ms_per_packet=0.3,
        dense_dispatch_ms=1.0,
        dense_launch_ms=2.0,
        subset_dispatch_ms=3.0,
        subset_launch_ms=4.0,
        packed_dispatch_ms=5.0,
        packed_launch_ms=6.0,
        subset_gather_ms_per_row=0.05,
    )
    work = cost.work_accounting(plan, ledger)
    assert work.candidate_nodes == len(tree.nodes)
    assert work.scored_candidate_source_rows == len(tree.nodes) * (3 * 3 + 2 * 2)
    assert work.active_frontier_nodes < work.candidate_nodes * 5
    assert work.unique_source_receiver_pairs == ledger.qm_unique_rows + ledger.qe_unique_rows
    assert work.dense_masked_executed_rows == ledger.qm_rectangular_rows + ledger.qe_rectangular_rows
    assert work.subset_executed_rows == work.packed_executed_rows == work.unique_source_receiver_pairs
    assert work.dense_masked_padded_rows == work.dense_masked_executed_rows - work.unique_source_receiver_pairs
    breakdown = cost.estimate_breakdown(plan, ledger, executor="rectangular_subset")
    assert breakdown["candidate_nodes"] == work.candidate_nodes
    assert breakdown["scored_candidate_source_rows"] == work.scored_candidate_source_rows
    assert breakdown["subset_executed_rows"] == work.subset_executed_rows
    legacy = MeasuredCostCoefficients("legacy-v1-fixture", 0, 1, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="cannot estimate typed plans"):
        legacy.estimate_ms(plan, ledger)


def test_bounded_oracle_accepts_only_protected_train_evidence_and_real_row_gain() -> None:
    """A synthetic observer exercises the gate; it is no physical label test."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tree, plan = _fixture(device)
    queries = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], device=device)
    module_present = torch.tensor([1.0, 1.0, 0.0], device=device)
    environment_weights = torch.tensor([2.0, 3.0], device=device)
    module_relevance = torch.ones_like(plan.module_membership)
    environment_relevance = torch.ones_like(plan.environment_membership)
    module_relevance[1, 1] = module_relevance[2, 0] = 0.0
    environment_relevance[1, 1] = environment_relevance[2, 0] = 0.0
    resolved_module = torch.zeros_like(module_relevance, dtype=torch.bool)
    resolved_environment = torch.zeros_like(environment_relevance, dtype=torch.bool)
    resolved_module[1, 1] = resolved_module[2, 0] = True
    resolved_environment[1, 1] = resolved_environment[2, 0] = True

    def observe(candidate: AdaptiveCoverPlan) -> OracleObservation:
        access = candidate.access(queries)
        needed = min(
            float(access.module_source[0, 0]), float(access.module_source[1, 1]),
            float(access.environment_source[0, 0]), float(access.environment_source[1, 1]),
        )
        return OracleObservation(
            roles={"critical_receiver": RoleError(0.02 if needed > 0.9 else 0.5, 0.01, True)},
            reference_evidence_id="synthetic-training-fixture",
            teacher_checkpoint_id="synthetic-teacher-fixture",
        )

    result = search_training_cover(
        plan, queries,
        module_present=module_present,
        environment_weights=environment_weights,
        observe=observe,
        propose=lambda current: local_cover_proposals(
            current,
            module_relevance=module_relevance,
            environment_relevance=environment_relevance,
            resolved_module=resolved_module,
            resolved_environment=resolved_environment,
            relevance_threshold=0.1,
            train_evidence_id="synthetic-training-fixture",
        ),
        limits={"critical_receiver": RoleLimit(0.1, 0.0)},
        costs=MeasuredCostCoefficients(
            "synthetic-timing-fixture", 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
        ),
        max_evaluations=6,
    )
    assert result.baseline_ledger.qm_unique_rows == 4
    assert result.selected_ledger.qm_unique_rows == 2
    assert result.selected_ledger.qe_unique_rows == 2
    assert result.selected_estimated_ms < result.baseline_estimated_ms
    assert result.selected_plan.access(tree.universe.coordinates).active_groups_on_anchors == 2
    assert any(trial.accepted for trial in result.trials)
    assert all(trial.proposal.train_evidence_id == "synthetic-training-fixture" for trial in result.trials)


@pytest.mark.parametrize(
    ("proposal_id", "candidate_reference", "candidate_teacher", "error"),
    (
        ("held", "train", "teacher", "proposal must use"),
        ("train", "held", "teacher", "observation changed its reference"),
        ("train", "train", "other-teacher", "observation changed its frozen teacher"),
    ),
)
def test_oracle_rejects_mixed_evidence_provenance(
    proposal_id: str, candidate_reference: str, candidate_teacher: str, error: str
) -> None:
    tree, plan = _fixture(torch.device("cpu"))
    observations = 0

    def observe(_candidate: AdaptiveCoverPlan) -> OracleObservation:
        nonlocal observations
        observations += 1
        return OracleObservation(
            roles={"receiver": RoleError(0.02, 0.02, True)},
            reference_evidence_id="train" if observations == 1 else candidate_reference,
            teacher_checkpoint_id="teacher" if observations == 1 else candidate_teacher,
        )

    with pytest.raises(ValueError, match=error):
        search_training_cover(
            plan,
            tree.universe.coordinates,
            module_present=torch.tensor([1.0, 1.0, 0.0]),
            environment_weights=torch.tensor([1.0, 1.0]),
            observe=observe,
            propose=lambda current: (
                OracleProposal("prune", 0, "module", 0, current, proposal_id),
            ),
            limits={"receiver": RoleLimit(0.1, 0.0)},
            costs=MeasuredCostCoefficients("synthetic", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            max_evaluations=1,
        )
