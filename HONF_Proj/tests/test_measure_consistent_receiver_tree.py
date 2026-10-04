"""Physical-measure equivalence through rebuilt trees and cost reductions."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.receiver_tree_access import build_receiver_tree_geometry, receiver_tree_access
from honf_forward_core.interface_fields.typed_hypergraph_state import (
    TypedHypergraphState,
    source_moments,
    structural_cost,
)


def _tree(coords, mass, roles=None, *, faithful=True, min_leaf=1):
    if roles is None:
        roles = torch.zeros(len(mass), dtype=torch.long)
    return CaseLocalReceiverTree.build(ReceiverAnchorUniverse(coords, mass, roles, coords.new_tensor([8., 4.])),
        max_nodes=15, min_leaf_anchors=min_leaf, max_depth=3, measure_consistent=faithful)


@pytest.mark.parametrize("coefficients", [[.5, .5], [.3, .7], [.1, .2, .7]])
def test_rebuild_unequal_refinement_permutation_and_coordinate_mass_gradients(coefficients):
    torch.manual_seed(43)
    base_coords = torch.tensor([[0., 0.], [1., 1.], [2., .4], [3., .7], [5., 0.]], dtype=torch.float64)
    base_mass = torch.tensor([.7, 2., 1.5, .2, .6], dtype=torch.float64)
    outputs, gradients, views = [], [], []
    for refined in (False, True):
        coords = base_coords.clone().requires_grad_()
        mass = base_mass.clone().requires_grad_()
        if refined:
            parents = torch.tensor([0] * len(coefficients) + [1, 2, 3, 4])
            fractions = mass.new_tensor(coefficients + [1.] * 4)
            order = torch.arange(len(parents) - 1, -1, -1)
            indexed_coords, indexed_mass = coords[parents][order], (mass[parents] * fractions)[order]
        else:
            indexed_coords, indexed_mass = coords, mass
        tree = _tree(indexed_coords, indexed_mass)
        views.append(tree.canonical_index)
        query = base_coords.new_tensor([[.5, .1], [1.4, .5], [2.7, .3], [4., .1]])
        gates = torch.linspace(.2, .8, 15, dtype=torch.float64)[None]
        data = build_receiver_tree_geometry([tree], 15)
        value = receiver_tree_access(query[None], gates, data, 3)
        scalar = tree.access(query, gates[0, :len(tree.nodes)])
        torch.testing.assert_close(value[0, :, :len(tree.nodes)], scalar, atol=1e-13, rtol=1e-13)
        weights = torch.arange(value.numel(), dtype=value.dtype).reshape_as(value)
        gradients.append(torch.autograd.grad((value * weights).sum(), (coords, mass)))
        outputs.append(value)
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-12, rtol=1e-12)
    for expected, measured in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(expected, measured, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(views[0].coordinates, views[1].coordinates, atol=0, rtol=0)
    torch.testing.assert_close(views[0].weights, views[1].weights, atol=1e-15, rtol=1e-15)


def test_coincident_roles_never_split_and_min_leaf_counts_support_blocks():
    coords = torch.tensor([[0., 0.], [0., 0.], [0., 0.], [2., 0.]])
    mass = torch.tensor([.2, .3, .5, 1.])
    roles = torch.tensor([0, 0, 1, 0])
    tree = _tree(coords, mass, roles)
    assert len(tree.canonical_index.block_atoms) == 3
    assert tree.canonical_index.atom_to_block.tolist() == [0, 0, 1, 2]
    for node in tree.nodes:
        if node.is_leaf:
            continue
        left = set(tree.nodes[node.left].anchor_indices)
        right = set(tree.nodes[node.right].anchor_indices)
        assert {0, 1, 2}.issubset(left) or {0, 1, 2}.issubset(right)
    assert len(_tree(coords, mass, roles, min_leaf=3).nodes) == 1
    coincident = _tree(torch.zeros(6, 2), torch.ones(6), torch.arange(6))
    assert len(coincident.nodes) == 1


@pytest.mark.parametrize("count", [3, 5, 9, 31, 192])
def test_symmetric_mass_cut_ties_survive_fp32_refinement_and_role_normalization(count):
    coords = torch.stack((torch.arange(count, dtype=torch.float32), torch.zeros(count)), -1)
    physical = torch.full((count,), 1. / 192.)
    original = _tree(coords, physical / physical.sum())
    parent = torch.tensor([0, 0, *range(1, count)])
    coefficients = physical.new_tensor([.3, .7, *([1.] * (count - 1))])
    child_mass = physical[parent] * coefficients
    rebuilt = _tree(coords[parent], child_mass / child_mass.sum())
    assert len(original.nodes) == len(rebuilt.nodes)
    for first, second in zip(original.nodes, rebuilt.nodes):
        expected = set(first.anchor_indices)
        actual = set(parent[list(second.anchor_indices)].tolist())
        assert actual == expected
        assert first.split_axis == second.split_axis
    query = coords[:min(count, 19)] + .1
    torch.testing.assert_close(original.access(query, physical.new_ones(len(original.nodes))),
                               rebuilt.access(query, physical.new_ones(len(rebuilt.nodes))), rtol=2e-5, atol=2e-6)


def test_legacy_unweighted_boundary_and_faithful_weighted_boundary_are_explicit():
    coords = torch.tensor([[-1., 0.], [-1., 0.], [0., 0.], [2., 0.]])
    mass = torch.tensor([.3, .7, 1., 2.])
    legacy, faithful = _tree(coords, mass, faithful=False), _tree(coords, mass)
    assert not legacy.measure_consistent
    assert faithful.measure_consistent
    assert faithful.split_boundary(0) == .75
    # Historical construction retains its unweighted child centroids.
    torch.testing.assert_close(legacy.split_boundary(0), torch.tensor(2. / 3.))


@pytest.mark.parametrize("faithful", [False, True])
def test_existing_plan_serialization_preserves_declared_boundary_semantics(faithful):
    tree = _tree(torch.tensor([[-1., 0.], [-1., 0.], [0., 0.], [2., 0.]]),
                 torch.tensor([.3, .7, 1., 2.]), faithful=faithful)
    plan = MechanismPlan.full_access(tree, torch.ones(2), 4).with_split(0, 1.)
    restored = MechanismPlan.from_dict(plan.to_dict())
    assert restored.tree.measure_consistent == faithful
    assert restored.canonical_hash() == plan.canonical_hash()
    torch.testing.assert_close(restored.tree.split_boundary(0), tree.split_boundary(0))
    if not faithful:
        assert "measure_consistent" not in plan.to_dict()["tree"]


def _cost_fixture(refined):
    dtype = torch.float64
    parent = torch.tensor([0, 0, 1, 2]) if refined else torch.arange(3)
    fraction = torch.tensor([.3, .7, 1., 1.], dtype=dtype) if refined else torch.ones(3, dtype=dtype)
    e_mass = torch.tensor([[.5, 2., 3.], [1., 4., .2]], dtype=dtype)[:, parent] * fraction
    m_mass = torch.tensor([[1., 1.], [1., 0.]], dtype=dtype)
    base_e = torch.tensor([[[.2, 1.4, .6], [1.8, .5, .3]], [[.3, .8, 1.6], [.5, 1.7, .4]]], dtype=dtype)
    member_e = base_e[:, :, parent]
    member_m = torch.tensor([[[.5, 1.5], [1.8, .2]], [[1., 0.], [.6, 0.]]], dtype=dtype)
    controls = torch.randn(2, 2, 3, dtype=dtype)
    state = TypedHypergraphState(memberships={"QE": member_e, "EM": member_m},
        controls={"QE": controls, "EM": controls}, centres=torch.zeros(2, 2, 2, dtype=dtype),
        admission=torch.tensor([[.6, .4], [.3, .7]], dtype=dtype),
        source_coords={"M": torch.zeros(2, 2, 2), "E": torch.zeros(2, len(parent), 2)},
        source_measures={"M": m_mass, "E": e_mass}, source_valid={"M": m_mass > 0, "E": e_mass > 0},
        source_ids={"M": torch.arange(2).expand(2, -1), "E": parent.expand(2, -1)},
        strategy_data={"measure_consistent": True,
                       "control_memberships": {tau: {"M": member_m, "E": member_e} for tau in ("QE", "EM")}})
    query_edge = torch.tensor([[[.2, .8], [.7, .3]], [[.4, .6], [.8, .2]]], dtype=dtype)
    env_edge = torch.tensor([[[.2, .8], [.3, .7], [.9, .1]], [[.3, .7], [.8, .2], [.5, .5]]], dtype=dtype)[:, parent]
    accesses = {"QE": source_moments(query_edge, member_e, controls, e_mass, e_mass > 0),
                "EM": source_moments(env_edge, member_m, controls, m_mass, m_mass > 0)}
    return state, accesses


def test_structural_cost_invariant_to_source_and_receiver_refinement_including_dual_donors():
    torch.manual_seed(3)
    original_state, original_access = _cost_fixture(False)
    torch.manual_seed(3)
    refined_state, refined_access = _cost_fixture(True)
    original, original_metrics = structural_cost(original_access, original_state)
    refined, refined_metrics = structural_cost(refined_access, refined_state)
    torch.testing.assert_close(original, refined, atol=1e-14, rtol=1e-14)
    for key in ("QE_smooth_pair_fraction", "EM_smooth_pair_fraction", "smooth_incidence_cost", "smooth_value_incidence_cost"):
        torch.testing.assert_close(original_metrics[key], refined_metrics[key], atol=1e-14, rtol=1e-14)
    assert refined_metrics["EM_control_pool_rows"] > original_metrics["EM_control_pool_rows"]


def test_measure_cost_averages_cases_then_mechanisms_and_honors_query_measure():
    state, accesses = _cost_fixture(False)
    mass = torch.tensor([[1., 3.], [2., 1.]], dtype=torch.float64)
    accesses["QE"].diagnostics["receiver_measures"] = mass
    _, metrics = structural_cost(accesses, state)
    density = accesses["QE"].density
    occupancy = 1 - torch.exp(-3 * density)
    weights = mass[..., None] * state.source_measures["E"][:, None]
    expected = ((occupancy * weights).sum((1, 2)) / weights.sum((1, 2))).mean()
    torch.testing.assert_close(metrics["QE_smooth_pair_fraction"], expected)
    legacy = replace(state, strategy_data={})
    _, legacy_metrics = structural_cost(accesses, legacy)
    torch.testing.assert_close(legacy_metrics["QE_smooth_pair_fraction"], occupancy.mean())


def test_measure_cost_refinement_gradient_pulls_back_to_original_membership_and_mass():
    state, initial = _cost_fixture(False)
    gradients = []
    for refined in (False, True):
        member = state.memberships["QE"].clone().requires_grad_()
        mass = state.source_measures["E"].clone().requires_grad_()
        parent = torch.tensor([0, 0, 1, 2]) if refined else torch.arange(3)
        fraction = mass.new_tensor([.3, .7, 1., 1.]) if refined else mass.new_ones(3)
        e_member, e_mass = member[:, :, parent], mass[:, parent] * fraction
        measures = {**state.source_measures, "E": e_mass}
        valid = {**state.source_valid, "E": e_mass > 0}
        memberships = {**state.memberships, "QE": e_member}
        donors = {tau: {"M": state.memberships["EM"], "E": e_member} for tau in memberships}
        revised = replace(state, source_measures=measures, source_valid=valid, memberships=memberships,
            strategy_data={"measure_consistent": True, "control_memberships": donors})
        accesses = {"QE": source_moments(initial["QE"].edge_access, e_member, state.controls["QE"], e_mass, e_mass > 0),
                    "EM": source_moments(initial["EM"].edge_access[:, parent], state.memberships["EM"],
                                         state.controls["EM"], measures["M"], valid["M"])}
        cost, _ = structural_cost(accesses, revised)
        gradients.append(torch.autograd.grad(cost, (member, mass)))
    for first, second in zip(*gradients):
        torch.testing.assert_close(first, second, atol=1e-14, rtol=1e-13)


def test_structural_policy_preserves_initial_mean_and_explicitly_versions_empty_mechanism_amendment():
    state, accesses = _cost_fixture(False)
    valid_m = state.source_valid["M"]
    controls = state.controls["EM"]
    edge = accesses["QE"].edge_access
    module_member = state.memberships["EM"]
    # An entirely ineligible self-only MM route must contribute zero under
    # the initial policy, and be excluded under the declared amendment.
    empty_mm = source_moments(edge, module_member, controls, state.source_measures["M"], valid_m,
        pair_valid=torch.zeros_like(accesses["EM"].density[:, :2], dtype=torch.bool))
    ordinary = {"QE": accesses["QE"], "EM": accesses["EM"]}
    mixed = {**ordinary, "MM": empty_mm}
    policy1 = replace(state, strategy_data={**state.strategy_data, "structural_measure_policy_version": 1})
    policy2 = replace(state, strategy_data={**state.strategy_data, "structural_measure_policy_version": 2})
    _, initial = structural_cost(mixed, policy1)
    _, amended = structural_cost(mixed, policy2)
    expected_initial = torch.stack([initial[f"{tau}_smooth_pair_fraction"] for tau in mixed]).mean()
    expected_amended = torch.stack([initial[f"{tau}_smooth_pair_fraction"] for tau in ordinary]).mean()
    torch.testing.assert_close(initial["smooth_pair_cost"], expected_initial, atol=0, rtol=0)
    torch.testing.assert_close(amended["smooth_pair_cost"], expected_amended, atol=0, rtol=0)
    for key in ("smooth_incidence_cost", "smooth_group_cost"):
        torch.testing.assert_close(initial[key], amended[key], atol=0, rtol=0)
    first, first_metrics = structural_cost(ordinary, policy1)
    second, second_metrics = structural_cost(ordinary, policy2)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first_metrics["smooth_pair_cost"], second_metrics["smooth_pair_cost"], atol=0, rtol=0)


def test_organizer_defaults_to_checkpoint_compatible_structural_policy():
    from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph

    from .test_adaptive_receiver_hypergraph import _case
    model = AdaptiveReceiverHypergraph(8, faithful_controls=True, measure_consistent=True).eval()
    encoded = _case()
    state = model.prepare(encoded, encoded.module_tokens)
    assert state.strategy_data["structural_measure_policy_version"] == 1
    assert state.export()["structural_measure_policy_version"] == 1
    for invalid in (0, 3, True):
        with pytest.raises(ValueError, match="policy version"):
            AdaptiveReceiverHypergraph(8, structural_measure_policy_version=invalid)
