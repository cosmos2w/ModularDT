from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.packet_controls import (
    canonical_packet_quotient,
    function_preserving_deeper_split,
    function_preserving_root_split,
    project_direct_pair_budget,
    root_union_plan,
)


def _tree(dtype: torch.dtype = torch.float64) -> CaseLocalReceiverTree:
    coordinates = torch.arange(16, dtype=dtype).reshape(-1, 1)
    universe = ReceiverAnchorUniverse(
        coordinates=coordinates,
        weights=torch.ones(16, dtype=dtype),
        roles=torch.zeros(16, dtype=torch.long),
        coordinate_scale=torch.ones(1, dtype=dtype),
    )
    return CaseLocalReceiverTree.build(
        universe,
        max_nodes=15,
        min_leaf_anchors=2,
        overlap_fraction=0.1,
    )


def _split_fixture() -> MechanismPlan:
    tree = _tree()
    node_count = len(tree.nodes)
    module_present = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    environment_count = 3
    split_gates = torch.zeros(node_count, dtype=torch.float64)
    split_gates[0] = 1.0
    # This gate is dormant below the closed target node. Opening the target
    # must reset it so the initial refinement consists of exactly two children.
    split_gates[3] = 1.0
    rows = torch.arange(node_count, dtype=torch.float64)
    modules = torch.stack(
        (
            rows.remainder(2),
            (rows + 1).remainder(2),
            rows.remainder(3).eq(0).to(torch.float64),
        ),
        dim=1,
    )
    environments = torch.stack(
        (
            rows.remainder(2),
            (rows + 1).remainder(2),
            rows.remainder(3).eq(1).to(torch.float64),
        ),
        dim=1,
    )
    winter_environment = torch.stack(
        (
            (rows + 1).remainder(2),
            rows.remainder(3).eq(0).to(torch.float64),
            rows.remainder(2),
        ),
        dim=1,
    )
    return MechanismPlan(
        tree,
        split_gates,
        module_present,
        environment_count,
        {
            "MM": modules,
            "QE": environments,
            "winter:QE": winter_environment,
        },
    )


def _context_accesses(
    plan: MechanismPlan,
    receivers: torch.Tensor,
) -> dict[tuple[str, str | None], torch.Tensor]:
    phases = (None, "winter")
    return {
        (mechanism, phase): plan.access_for(mechanism, receivers, phase=phase)
        for mechanism in INTERACTION_MECHANISMS
        for phase in phases
    }


def _on_device(plan: MechanismPlan, device: torch.device) -> MechanismPlan:
    universe = replace(
        plan.tree.universe,
        coordinates=plan.tree.universe.coordinates.to(device),
        weights=plan.tree.universe.weights.to(device),
        roles=plan.tree.universe.roles.to(device),
        coordinate_scale=plan.tree.universe.coordinate_scale.to(device),
    )
    tree = replace(plan.tree, universe=universe)
    return MechanismPlan(
        tree,
        plan.split_gates.to(device),
        plan.module_present.to(device),
        plan.environment_count,
        {key: value.to(device) for key, value in plan.permissions.items()},
    )


def test_deeper_split_copies_actions_and_preserves_all_typed_access() -> None:
    plan = _split_fixture()
    receivers = plan.tree.universe.coordinates.clone()
    original_access = _context_accesses(plan, receivers)
    original_weights = plan.tree.universe.weights.clone()
    original_receivers = receivers.clone()

    refined = function_preserving_deeper_split(plan, node_index=1)

    assert refined.tree is plan.tree
    assert torch.equal(refined.tree.universe.weights, original_weights)
    assert torch.equal(receivers, original_receivers)
    assert float(refined.split_gates[1]) == 1.0
    assert float(refined.split_gates[3]) == 0.0
    assert float(refined.split_gates[4]) == 0.0
    for key, original in plan.permissions.items():
        for child in (plan.tree.nodes[1].left, plan.tree.nodes[1].right):
            assert child is not None
            assert torch.equal(refined.permissions[key][child], original[1])
    for context, expected in original_access.items():
        actual = refined.access_for(
            context[0],
            receivers,
            phase=context[1],
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_deeper_split_keeps_child_score_gradients_independent() -> None:
    plan = _split_fixture()
    split_gates = plan.split_gates.detach().clone().requires_grad_()
    permissions = {
        key: value.detach().clone().requires_grad_()
        for key, value in plan.permissions.items()
    }
    differentiable = MechanismPlan(
        plan.tree,
        split_gates,
        plan.module_present,
        plan.environment_count,
        permissions,
    )

    refined = function_preserving_deeper_split(differentiable, node_index=1)
    children = (plan.tree.nodes[1].left, plan.tree.nodes[1].right)
    assert children[0] is not None and children[1] is not None
    assert refined.permissions[next(iter(permissions))][children[0], 0].detach().item() == (
        refined.permissions[next(iter(permissions))][children[1], 0].detach().item()
    )
    child_terms = (
        refined.permissions[next(iter(permissions))][children[0], 0]
        + 2.0 * refined.permissions[next(iter(permissions))][children[1], 0]
        + refined.split_gates[1]
    )
    gradients = torch.autograd.grad(child_terms, permissions[next(iter(permissions))], retain_graph=True)[0]
    assert float(gradients[children[0], 0]) == 1.0
    assert float(gradients[children[1], 0]) == 2.0


def test_root_bootstrap_then_deeper_split_preserves_all_typed_access() -> None:
    fixture = _split_fixture()
    closed_root = MechanismPlan(
        fixture.tree,
        torch.zeros_like(fixture.split_gates),
        fixture.module_present,
        fixture.environment_count,
        fixture.permissions,
    )
    receivers = closed_root.tree.universe.coordinates.clone()
    original = _context_accesses(closed_root, receivers)

    root_refined = function_preserving_root_split(closed_root)
    deeper_refined = function_preserving_deeper_split(root_refined, node_index=1)

    assert float(root_refined.split_gates[0]) == 1.0
    assert float(deeper_refined.split_gates[1]) == 1.0
    for context, expected in original.items():
        torch.testing.assert_close(
            root_refined.access_for(context[0], receivers, phase=context[1]),
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            deeper_refined.access_for(context[0], receivers, phase=context[1]),
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_exact_quotient_combines_only_identical_full_typed_actions() -> None:
    plan = _split_fixture()
    gates = plan.split_gates.clone()
    gates[1] = 1.0
    gates[3] = 0.0
    modules = plan.permissions[next(key for key in plan.permissions if key.mechanism == "MM")].clone()
    environments = plan.permissions[next(key for key in plan.permissions if key.mechanism == "QE")].clone()
    winter = plan.permissions[next(key for key in plan.permissions if key.phase == "winter")].clone()
    # Active nodes 3 and 4 have exactly identical actions. Node 2 differs by a
    # single float64 step-sized value and must remain a distinct exact class.
    modules[3] = torch.tensor([1.0, 0.0, 1.0], dtype=modules.dtype)
    modules[4] = modules[3]
    modules[2] = modules[3]
    modules[2, 0] = 1.0 - 1.0e-12
    environments[3] = torch.tensor([1.0, 0.0, 1.0], dtype=environments.dtype)
    environments[4] = environments[3]
    environments[2] = environments[3]
    winter[3] = torch.tensor([0.0, 1.0, 0.0], dtype=winter.dtype)
    winter[4] = winter[3]
    winter[2] = winter[3]
    permissions = dict(plan.permissions)
    permissions[next(key for key in permissions if key.mechanism == "MM")] = modules
    permissions[next(key for key in permissions if key.mechanism == "QE")] = environments
    permissions[next(key for key in permissions if key.phase == "winter")] = winter
    plan = MechanismPlan(plan.tree, gates, plan.module_present, plan.environment_count, permissions)
    receivers = plan.tree.universe.coordinates.clone()

    quotient = canonical_packet_quotient(plan, receivers)

    assert (3, 4) in quotient.node_groups
    assert not any(2 in group and 3 in group for group in quotient.node_groups)
    for context, expected in _context_accesses(plan, receivers).items():
        actual = quotient.access_for(context[0], phase=context[1])
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_root_union_uses_global_frontier_independent_of_evaluation_panel() -> None:
    plan = function_preserving_deeper_split(_split_fixture(), node_index=1)
    left_child = plan.tree.nodes[1].left
    right_child = plan.tree.nodes[1].right
    assert left_child is not None and right_child is not None
    mm_key = next(key for key in plan.permissions if key.mechanism == "MM")
    modules = plan.permissions[mm_key].clone()
    modules[left_child] = torch.tensor([1.0, 0.0, 0.0], dtype=modules.dtype)
    modules[right_child] = torch.tensor([0.0, 0.0, 1.0], dtype=modules.dtype)
    modules[2] = torch.zeros(3, dtype=modules.dtype)
    permissions = dict(plan.permissions)
    permissions[mm_key] = modules
    plan = MechanismPlan(
        plan.tree,
        plan.split_gates,
        plan.module_present,
        plan.environment_count,
        permissions,
    )
    sampled_panel = plan.tree.universe.coordinates[[0]]
    other_native_receiver = plan.tree.universe.coordinates[[7]]
    sampled_access = plan.tree.access(sampled_panel, plan.split_gates)
    other_access = plan.tree.access(other_native_receiver, plan.split_gates)
    assert float(sampled_access[0, right_child]) == 0.0
    assert float(other_access[0, right_child]) > 0.0

    merged = root_union_plan(plan)

    assert merged.tree is plan.tree
    assert torch.equal(merged.split_gates, torch.zeros_like(plan.split_gates))
    assert set(merged.permissions) == set(plan.permissions)
    assert merged.explicit_bypass_keys == plan.explicit_bypass_keys
    assert merged.explicit_phase_keys == plan.explicit_phase_keys
    frontier = (2, left_child, right_child)
    for key, original in plan.permissions.items():
        expected = (
            (plan.permission_matrix(key.mechanism, phase=key.phase)[list(frontier)] > 0)
            .any(dim=0)
            .to(dtype=original.dtype)
        )
        assert torch.equal(merged.permissions[key][0], expected)
        assert torch.count_nonzero(merged.permissions[key][1:]) == 0
    assert float(merged.permissions[mm_key][0, 2]) == 1.0
    for mechanism in INTERACTION_MECHANISMS:
        full_union = merged.access_for(mechanism, sampled_panel)
        torch.testing.assert_close(
            full_union,
            full_union[:1].expand_as(full_union),
            rtol=0.0,
            atol=0.0,
        )


def test_root_union_rejects_soft_or_differentiable_plans() -> None:
    plan = _split_fixture()
    soft_gates = plan.split_gates.clone()
    soft_gates[0] = 0.5
    soft = MechanismPlan(
        plan.tree,
        soft_gates,
        plan.module_present,
        plan.environment_count,
        plan.permissions,
    )
    with pytest.raises(ValueError, match="hard binary split gates"):
        root_union_plan(soft)

    differentiable = MechanismPlan(
        plan.tree,
        plan.split_gates.detach().clone().requires_grad_(),
        plan.module_present,
        plan.environment_count,
        plan.permissions,
    )
    with pytest.raises(ValueError, match="detached evaluation plan"):
        root_union_plan(differentiable)


def test_direct_pair_budget_is_exact_and_ties_are_stable() -> None:
    scores = torch.tensor(
        [[0.2, 0.9], [0.9, 1.0]],
        dtype=torch.float64,
    )
    result = project_direct_pair_budget(scores, requested_work=2)

    expected = torch.tensor([[False, True], [False, True]])
    assert torch.equal(result.selected_pairs, expected)
    assert result.requested_work == 2
    assert result.achieved_work == 2
    assert result.feasible_capacity == 4
    assert result.exact_match
    assert result.closest_feasible_work == 2


def test_direct_pair_budget_reports_closest_capacity_and_respects_eligibility() -> None:
    scores = torch.tensor(
        [[0.1, 100.0, 0.3], [0.4, 0.5, 0.6]],
        dtype=torch.float64,
    )
    eligible = torch.tensor(
        [[True, False, True], [False, True, True]],
    )

    result = project_direct_pair_budget(
        scores,
        requested_work=5,
        eligible_pairs=eligible,
    )

    assert result.feasible_capacity == 4
    assert result.achieved_work == 4
    assert result.closest_feasible_work == 4
    assert result.work_shortfall == 1
    assert not result.exact_match
    assert torch.equal(result.selected_pairs, eligible)


def test_direct_pair_budget_validates_score_and_budget_contract() -> None:
    with pytest.raises(ValueError, match="finite"):
        project_direct_pair_budget(torch.tensor([[float("nan")]]), requested_work=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        project_direct_pair_budget(torch.ones((1, 1)), requested_work=-1)
    with pytest.raises(TypeError, match="integer"):
        project_direct_pair_budget(torch.ones((1, 1)), requested_work=1.5)  # type: ignore[arg-type]


@pytest.mark.skipif(
    os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "2" or not torch.cuda.is_available(),
    reason="CUDA integration requires the granted physical GPU2 slot as logical cuda:0",
)
def test_packet_controls_cuda_split_quotient_and_structural_root_union() -> None:
    device = torch.device("cuda:0")
    assert torch.cuda.current_device() == 0
    fixture = _split_fixture()
    closed_root = MechanismPlan(
        fixture.tree,
        torch.zeros_like(fixture.split_gates),
        fixture.module_present,
        fixture.environment_count,
        fixture.permissions,
    )
    plan = _on_device(closed_root, device)
    receivers = plan.tree.universe.coordinates
    original = _context_accesses(plan, receivers)

    refined = function_preserving_deeper_split(
        function_preserving_root_split(plan),
        node_index=1,
    )
    quotient = canonical_packet_quotient(refined, receivers)
    merged = root_union_plan(refined)

    assert refined.split_gates.device == device
    assert quotient.receiver_access.device == device
    assert merged.split_gates.device == device
    for context, expected in original.items():
        torch.testing.assert_close(
            refined.access_for(context[0], receivers, phase=context[1]),
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            quotient.access_for(context[0], phase=context[1]),
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    assert torch.equal(merged.split_gates, torch.zeros_like(merged.split_gates))
