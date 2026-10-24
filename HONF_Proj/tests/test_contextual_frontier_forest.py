"""Focused tests for contextual candidate forests and learned frontiers."""
from __future__ import annotations

import copy

import pytest
import torch

from honf_forward_core.contextual_organization.forest import (
    ContextualFrontierPolicy,
    SceneForest,
    build_forest,
)


def _scene(count: int = 5, *, requires_grad: bool = False):
    ids = torch.tensor([41, 7, 23, 5, 18, 52, 2][:count], dtype=torch.long)
    states = torch.tensor(
        [[0.3, 0.1, 0.8, -0.2], [0.7, -0.4, 0.1, 0.3], [0.2, 0.9, -0.1, 0.4],
         [-0.3, 0.4, 0.7, 0.8], [0.6, 0.2, -0.5, 0.1], [0.1, -0.6, 0.4, 0.2],
         [-0.7, 0.5, 0.1, -0.3]][:count], dtype=torch.float64,
    )
    coordinates = torch.stack(
        (torch.arange(count, dtype=torch.float64), torch.zeros(count, dtype=torch.float64)), dim=-1
    )
    lengths = torch.full_like(coordinates, 0.5)
    measure = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    return (
        states.requires_grad_(requires_grad),
        coordinates.requires_grad_(requires_grad),
        lengths,
        measure,
        ids,
    )


def _build(*, count: int = 5, requires_grad: bool = False, bank: str = "E", k: int = 4):
    states, coordinates, lengths, measure, ids = _scene(count, requires_grad=requires_grad)
    forest = build_forest(states, coordinates, lengths, measure, ids, bank, k=k)
    return forest


def _merge_signature(forest: SceneForest):
    return tuple(node.ancestry_ids for node in forest.nodes if not node.is_leaf)


def _assert_exact_cover(forest: SceneForest, frontier):
    flattened = [row for ancestry in frontier.ancestry_local_indices for row in ancestry]
    assert sorted(flattened) == list(range(forest.record_count))
    assert len(flattened) == forest.record_count


def test_forest_is_full_binary_and_keeps_original_ancestry_and_spatial_halo():
    forest = _build(count=6, k=2)
    assert forest.root is not None
    assert len(forest.nodes) == 2 * forest.record_count - 1
    assert len(forest.internal_indices) == forest.record_count - 1
    assert forest.nodes[forest.root].ancestry_local_indices == tuple(
        sorted(range(6), key=lambda row: forest.original_ids[row])
    )
    for row, original_id in enumerate(forest.original_ids):
        leaf = forest.nodes[row]
        assert leaf.is_leaf
        assert leaf.ancestry_local_indices == (row,)
        assert leaf.ancestry_ids == (original_id,)
    for node in forest.nodes:
        if node.children is None:
            continue
        left, right = node.children
        expected = tuple(
            sorted(
                forest.nodes[left].ancestry_local_indices + forest.nodes[right].ancestry_local_indices,
                key=lambda row: forest.original_ids[row],
            )
        )
        assert node.ancestry_local_indices == expected

    assert forest.stats.candidates == 2 * forest.record_count - 1
    assert forest.stats.spatial_spanning_edges == forest.record_count - 1
    assert len(forest.spatial_neighbors) == forest.record_count
    assert all(len(neighbors) <= 4 for neighbors in forest.spatial_neighbors)
    assert forest.stats.topology_detached


def test_context_knn_changes_with_context_and_stays_out_of_spatial_halo():
    states, coordinates, lengths, measure, ids = _scene(6)
    first_states = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    second_states = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    first = build_forest(first_states, coordinates, lengths, measure, ids, "E", k=1)
    second = build_forest(second_states, coordinates, lengths, measure, ids, "E", k=1)
    assert first.context_neighbors != second.context_neighbors
    context_only = [edge for edge in first.candidate_edges if edge.context_knn and not edge.spatial_knn]
    assert context_only
    for edge in context_only:
        assert edge.right not in first.spatial_neighbors[edge.left]
        assert edge.left not in first.spatial_neighbors[edge.right]


def test_id_ties_make_topology_invariant_to_input_row_permutation():
    states = torch.zeros((4, 3), dtype=torch.float64)
    coordinates = torch.zeros((4, 2), dtype=torch.float64)
    lengths = torch.ones_like(coordinates)
    measure = torch.ones((4,), dtype=torch.float64)
    ids = torch.tensor([30, 10, 40, 20], dtype=torch.long)
    first = build_forest(states, coordinates, lengths, measure, ids, "M", k=1)
    order = torch.tensor([2, 0, 3, 1])
    second = build_forest(
        states[order], coordinates[order], lengths[order], measure[order], ids[order], "M", k=1
    )
    assert _merge_signature(first) == _merge_signature(second)
    first_edges = {
        tuple(sorted((first.original_ids[edge.left], first.original_ids[edge.right])))
        for edge in first.candidate_edges
    }
    second_edges = {
        tuple(sorted((second.original_ids[edge.left], second.original_ids[edge.right])))
        for edge in second.candidate_edges
    }
    assert first_edges == second_edges


def test_node_pooling_and_policy_risk_keep_real_tensor_gradients():
    forest = _build(requires_grad=True)
    policy = ContextualFrontierPolicy(4, 2, risk_dim=2)
    frontier = policy(forest, epoch=21, case_key="case-a", update=3, training=True)
    assert frontier.expected_K.requires_grad
    assert frontier.log_prob.requires_grad
    risk_loss = torch.stack([value.square().sum() for value in frontier.risk_logits.values()]).sum()
    risk_loss.backward()
    assert forest.states.grad is not None
    assert float(forest.states.grad.abs().sum()) > 0.0
    assert policy.risk_head.weight.grad is not None
    assert float(policy.risk_head.weight.grad.abs().sum()) > 0.0


def test_batched_policy_scores_match_per_node_values_and_gradients():
    forest = _build(requires_grad=True)
    policy = ContextualFrontierPolicy(4, 2, risk_dim=2).double()
    stop, risk = policy._node_scores(forest)
    manual_stop = {}
    manual_risk = {}
    for node in forest.nodes:
        if node.children is None:
            continue
        left, right = (forest.nodes[index].descriptor for index in node.children)
        descriptor = node.descriptor.to(dtype=policy.shared_scorer[0].weight.dtype)
        left = left.to(dtype=policy.shared_scorer[0].weight.dtype)
        right = right.to(dtype=policy.shared_scorer[0].weight.dtype)
        features = torch.cat((descriptor, 0.5 * (left + right), torch.abs(left - right)), dim=-1)
        shared = policy.shared_scorer(features)
        manual_stop[node.index] = policy.stop_head(shared).reshape(())
        manual_risk[node.index] = policy.risk_head(shared)

    assert stop.keys() == manual_stop.keys()
    assert risk.keys() == manual_risk.keys()
    for index in stop:
        torch.testing.assert_close(stop[index], manual_stop[index], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(risk[index], manual_risk[index], rtol=1e-10, atol=1e-10)

    batched_loss = torch.stack([value.square() for value in stop.values()]).sum()
    batched_loss = batched_loss + torch.stack([value.square().sum() for value in risk.values()]).sum()
    manual_loss = torch.stack([value.square() for value in manual_stop.values()]).sum()
    manual_loss = manual_loss + torch.stack([value.square().sum() for value in manual_risk.values()]).sum()
    parameters = tuple(policy.parameters()) + (forest.states,)
    batched_gradients = torch.autograd.grad(batched_loss, parameters, retain_graph=True, allow_unused=True)
    manual_gradients = torch.autograd.grad(manual_loss, parameters, allow_unused=True)
    for batched, manual in zip(batched_gradients, manual_gradients):
        if batched is None or manual is None:
            assert batched is manual is None
        else:
            torch.testing.assert_close(batched, manual, rtol=1e-10, atol=1e-10)


def test_warmup_forces_leaves_and_expected_k_has_analytic_gradient():
    forest = _build(count=4)
    policy = ContextualFrontierPolicy(4, 2, run_seed=17)
    assert torch.all(policy.stop_head.bias == -4.0)
    assert torch.all(policy.stop_head.weight == 0.0)
    assert policy.temperature_for_epoch(1) == pytest.approx(2.0)
    assert policy.temperature_for_epoch(200) == pytest.approx(0.5)
    assert policy.temperature_for_epoch(500) == pytest.approx(0.5)
    assert policy.structural_multiplier_for_epoch(1) == pytest.approx(0.1)
    assert policy.structural_multiplier_for_epoch(200) == pytest.approx(1.0)

    frontier = policy(forest, epoch=20, case_key="case-a", update=5, training=True)
    assert frontier.K == forest.record_count
    assert frontier.mode == "forced_leaves"
    assert not frontier.capacity_active
    assert not frontier.score_function_active
    _assert_exact_cover(forest, frontier)
    frontier.expected_K.backward()
    assert policy.stop_head.bias.grad is not None
    assert float(policy.stop_head.bias.grad.abs().sum()) > 0.0


def test_enumerated_frontiers_have_reached_only_log_probability_and_exact_score_gradient():
    forest = _build(count=3)
    policy = ContextualFrontierPolicy(4, 2, run_seed=9, hidden_dim=12).double()
    root = forest.root
    assert root is not None
    child_internal = next(child for child in forest.nodes[root].children if not forest.nodes[child].is_leaf)

    choices = (
        ({root: True}, 0.2),
        ({root: False, child_internal: True}, 1.1),
        ({root: False, child_internal: False}, -0.8),
    )
    probabilities = []
    log_probabilities = []
    losses = []
    for decisions, loss in choices:
        frontier = policy(
            forest,
            epoch=21,
            case_key="case-enumeration",
            update=2,
            training=True,
            forced_decisions=decisions,
        )
        assert set(frontier.decisions) == set(decisions)
        assert frontier.log_prob.requires_grad
        _assert_exact_cover(forest, frontier)
        probabilities.append(frontier.log_prob.exp())
        log_probabilities.append(frontier.log_prob)
        losses.append(loss)
    probability_sum = torch.stack(probabilities).sum()
    torch.testing.assert_close(probability_sum, torch.ones_like(probability_sum))

    exact_objective = torch.stack(
        [probability * loss for probability, loss in zip(probabilities, losses)]
    ).sum()
    baseline = 0.37
    score_objective = torch.stack(
        [probability.detach() * (loss - baseline) * log_probability
         for probability, loss, log_probability in zip(probabilities, losses, log_probabilities)]
    ).sum()
    parameters = (policy.stop_head.weight, policy.stop_head.bias)
    exact_gradient = torch.autograd.grad(exact_objective, parameters, retain_graph=True)
    score_gradient = torch.autograd.grad(score_objective, parameters)
    for exact, score in zip(exact_gradient, score_gradient):
        torch.testing.assert_close(score, exact, rtol=1e-10, atol=1e-10)


def test_keyed_samples_are_row_order_and_call_order_independent_and_restore():
    forest = _build(count=6)
    states, coordinates, lengths, measure, ids = _scene(6)
    order = torch.tensor([4, 0, 5, 2, 1, 3])
    permuted = build_forest(
        states[order], coordinates[order], lengths[order], measure[order], ids[order], "E", k=4
    )
    policy = ContextualFrontierPolicy(4, 2, run_seed=701)
    first = policy(forest, epoch=21, case_key="train-0138", update=7, training=True)
    _ = policy(forest, epoch=21, case_key="another-case", update=8, training=True)
    second = policy(permuted, epoch=21, case_key="train-0138", update=7, training=True)
    first_decisions = {
        tuple(forest.nodes[index].identity): decision for index, decision in first.decisions.items()
    }
    second_decisions = {
        tuple(permuted.nodes[index].identity): decision for index, decision in second.decisions.items()
    }
    assert first_decisions == second_decisions
    assert {
        tuple(sorted(ids_in_node)) for ids_in_node in first.ancestry_ids
    } == {tuple(sorted(ids_in_node)) for ids_in_node in second.ancestry_ids}

    state = copy.deepcopy(policy.state_dict())
    restored = ContextualFrontierPolicy(4, 2, run_seed=0)
    restored.load_state_dict(state)
    assert restored.run_seed == 701
    replay = restored(forest, epoch=21, case_key="train-0138", update=7, training=True)
    assert {
        tuple(forest.nodes[index].identity): decision for index, decision in replay.decisions.items()
    } == first_decisions
    assert replay.sampling_identity == first.sampling_identity


def test_forced_subtree_changes_only_its_ancestry_and_never_breaks_cover():
    forest = _build(count=6)
    policy = ContextualFrontierPolicy(4, 2, run_seed=77)
    baseline = policy(forest, epoch=21, case_key="case-int", update=11, training=False)
    target = next(index for index in forest.internal_indices if index != forest.root)
    coarse = policy(
        forest,
        epoch=21,
        case_key="case-int",
        update=11,
        training=False,
        forced_subtree=(target, "coarse"),
    )
    _assert_exact_cover(forest, baseline)
    _assert_exact_cover(forest, coarse)
    target_members = set(forest.nodes[target].ancestry_local_indices)
    baseline_outside = {
        index for node_index in baseline.node_indices for index in forest.nodes[node_index].ancestry_local_indices
        if index not in target_members
    }
    coarse_outside = {
        index for node_index in coarse.node_indices for index in forest.nodes[node_index].ancestry_local_indices
        if index not in target_members
    }
    assert baseline_outside == coarse_outside
    assert target in coarse.node_indices


def test_forced_subtree_intervention_overrides_jd_leaf_policy_only_on_target_path():
    forest = _build(count=6)
    policy = ContextualFrontierPolicy(4, 2, run_seed=77)
    target = next(index for index in forest.internal_indices if index != forest.root)
    leaves = policy(
        forest,
        epoch=21,
        case_key="case-int-jd",
        update=11,
        training=True,
        force_leaves=True,
    )
    coarse = policy(
        forest,
        epoch=21,
        case_key="case-int-jd",
        update=11,
        training=True,
        force_leaves=True,
        forced_subtree=(target, "coarse"),
    )
    fine = policy(
        forest,
        epoch=21,
        case_key="case-int-jd",
        update=11,
        training=True,
        force_leaves=True,
        forced_subtree=(target, "split"),
    )

    for frontier in (leaves, coarse, fine):
        _assert_exact_cover(forest, frontier)
    assert leaves.K == forest.record_count
    assert target in coarse.node_indices
    assert target not in fine.node_indices
    target_members = set(forest.nodes[target].ancestry_local_indices)
    for frontier in (coarse, fine):
        outside = {
            row
            for node_index in frontier.node_indices
            for row in forest.nodes[node_index].ancestry_local_indices
            if row not in target_members
        }
        leaf_outside = {
            row
            for node_index in leaves.node_indices
            for row in forest.nodes[node_index].ancestry_local_indices
            if row not in target_members
        }
        assert outside == leaf_outside
    assert coarse.K < fine.K
    assert not coarse.score_function_active and not fine.score_function_active


def test_empty_and_singleton_banks_have_valid_covers():
    empty_states = torch.empty((0, 4), dtype=torch.float32)
    empty_coordinates = torch.empty((0, 2), dtype=torch.float32)
    empty = build_forest(
        empty_states, empty_coordinates, empty_coordinates, torch.empty((0,)), [], "M"
    )
    policy = ContextualFrontierPolicy(4, 2)
    empty_frontier = policy(empty, epoch=1, case_key="empty", training=False)
    assert empty.root is None
    assert empty_frontier.K == 0
    assert empty_frontier.expected_K.item() == 0.0
    assert empty_frontier.node_indices == ()

    singleton_states = torch.ones((1, 4), dtype=torch.float32)
    singleton_coordinates = torch.zeros((1, 2), dtype=torch.float32)
    singleton = build_forest(
        singleton_states,
        singleton_coordinates,
        torch.ones_like(singleton_coordinates),
        torch.zeros((1,), dtype=torch.float32),
        ["present-zero-measure"],
        "E",
    )
    singleton_frontier = policy(singleton, epoch=1, case_key="one", training=False)
    assert singleton.root == 0
    assert singleton.nodes[0].mass.item() == 0.0
    assert singleton_frontier.K == 1
    assert singleton_frontier.ancestry_ids == (("present-zero-measure",),)
