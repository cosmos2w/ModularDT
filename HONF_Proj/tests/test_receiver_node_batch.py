"""Batched node reductions preserve the previous formulas and first VJPs."""

from copy import deepcopy
from dataclasses import replace
from types import MethodType

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CandidateNode,
    CaseLocalReceiverTree,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph, _pool
from honf_forward_core.interface_fields.typed_hypergraph_state import pad_geometry
from honf_forward_core.interface_fields.types import EncodedInterfaceCase

from ._receiver_case_loop_reference import case_loop_prepare


def _scalar_key_reference(universe, capacity, minimum, depth_limit):
    """Historical host sort/cut arithmetic, including tensor scalar keys."""
    coordinates = (universe.coordinates / universe.coordinate_scale).detach().cpu()
    weights, roles = universe.weights.detach().cpu(), universe.roles.detach().cpu()
    nodes, saturated = [CandidateNode(tuple(range(coordinates.shape[0])))], False

    def refine(index, indices, depth):
        nonlocal saturated
        if len(indices) <= minimum or depth >= depth_limit:
            return
        if len(nodes) + 2 > capacity:
            saturated = True
            return
        subset = coordinates[list(indices)]
        extent = subset.max(0).values - subset.min(0).values
        axis = int(torch.argmax(extent).item())
        if float(extent[axis]) <= 0:
            return
        ordered = sorted(indices, key=lambda item: (
            float(coordinates[item, axis]),
            *(float(value) for value in coordinates[item]), int(roles[item])))
        half_mass = float(weights[list(ordered)].sum()) / 2
        accumulated, cut = 0., 1
        for position, anchor in enumerate(ordered[:-1], start=1):
            accumulated += float(weights[anchor])
            cut = position
            if accumulated >= half_mass:
                break
        left_ids, right_ids = tuple(ordered[:cut]), tuple(ordered[cut:])
        left, right = len(nodes), len(nodes) + 1
        nodes.extend((CandidateNode(left_ids), CandidateNode(right_ids)))
        nodes[index] = CandidateNode(indices, left, right, axis)
        refine(left, left_ids, depth + 1)
        refine(right, right_ids, depth + 1)

    refine(0, nodes[0].anchor_indices, 0)
    return tuple(nodes), saturated


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_host_list_sort_preserves_weighted_tree_identity_and_access(dimension, dtype):
    torch.manual_seed(2304)
    # Repeated coordinates exercise role ties and stable source ordering;
    # unequal weights exercise the unchanged tensor half-mass reduction.
    coordinates = torch.randint(0, 4, (31, dimension)).to(dtype)
    coordinates[1] = coordinates[0]
    roles = torch.randint(0, 8, (31,))
    roles[1] = roles[0]
    mass = torch.rand(31, dtype=dtype) + .01
    universe = ReceiverAnchorUniverse(coordinates, mass, roles, torch.arange(1, dimension + 1, dtype=dtype))
    for capacity, minimum, depth in ((15, 1, 3), (9, 2, 5), (127, 1, 5)):
        expected, saturated = _scalar_key_reference(universe, capacity, minimum, depth)
        actual = CaseLocalReceiverTree.build(universe, max_nodes=capacity, min_leaf_anchors=minimum, max_depth=depth)
        assert actual.nodes == expected
        assert actual.capacity_saturated == saturated
        reference = CaseLocalReceiverTree(universe, expected, actual.overlap_fraction, saturated)
        query = torch.rand(17, dimension, dtype=dtype) * 4
        gates = torch.rand(len(expected), dtype=dtype)
        torch.testing.assert_close(actual.access(query, gates), reference.access(query, gates), rtol=0, atol=0)


def _unbatched_index(self, encoded, states, case, tau, case_scale, summaries, global_token, measures, phase_features, fixed_tree=None, *, encode_nodes=True):
    """Reference reduction order before node batching, retained for parity."""
    assert fixed_tree is None
    coords, mass, receiver = self._catalogue(encoded, self._continuous_input(states), case, tau)
    roles = self._receiver_roles(encoded, case, tau, coords.shape[0])
    tree = CaseLocalReceiverTree.build(ReceiverAnchorUniverse(coords, mass, roles, case_scale),
        max_nodes=self.capacity, min_leaf_anchors=1, overlap_fraction=self.overlap_fraction, max_depth=self.max_depth)
    centres, descriptors, depths = [], [], []
    node_depths = {0: 0}
    for index, node in enumerate(tree.nodes):
        ids = list(node.anchor_indices)
        node_mass = mass[ids]
        centre = _pool(coords[ids], node_mass)
        extent = coords[ids].amax(0) - coords[ids].amin(0)
        summary = _pool(receiver[ids], node_mass)
        role_summary = _pool(F.one_hot(roles[ids], 8).to(states.dtype), node_mass)
        statistics = torch.stack((measures["M"][case].sum().log1p(), measures["E"][case].sum().log1p(),
            (measures["M"][case].sum() > 0).to(states.dtype), (measures["E"][case].sum() > 0).to(states.dtype)))
        depth = node_depths[index]
        descriptors.append(torch.cat((summary, summaries["M"][case], summaries["E"][case], global_token[case],
            pad_geometry(centre / case_scale), pad_geometry(extent / case_scale), statistics,
            centre.new_tensor([depth / max(self.max_depth, 1)]), phase_features, role_summary)))
        centres.append(centre)
        depths.append(depth)
        if node.left is not None:
            node_depths[node.left] = depth + 1
            node_depths[node.right] = depth + 1
    descriptors = torch.stack(descriptors)
    return tree, torch.stack(centres), self.node_encoder(descriptors) if encode_nodes else descriptors, depths


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("soft", [False, True])
def test_node_batch_output_physical_input_and_parameter_first_gradient_parity(dimension, soft):
    torch.manual_seed(804)
    encoded = EncodedInterfaceCase(torch.randn(2, 4, 8, dtype=torch.float64), torch.randn(2, 12, 8, dtype=torch.float64),
        torch.randn(2, 8, dtype=torch.float64), torch.rand(2, 4, dimension, dtype=torch.float64),
        torch.rand(2, 12, dimension, dtype=torch.float64), torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 1.]], dtype=torch.float64),
        torch.randn(2, 4, 3, dtype=torch.float64), None, torch.rand(2, 12, dtype=torch.float64) + .1,
        torch.ones(dimension, dtype=torch.float64))
    batched = AdaptiveReceiverHypergraph(8).double().eval()
    with torch.no_grad():
        for head in (*batched.control_heads.values(), *batched.source_scores.values()):
            head[-1].weight.normal_(0, .1)
        batched.split_head.weight.normal_(0, .1)
    previous = deepcopy(batched)
    previous._index = MethodType(_unbatched_index, previous)
    outcomes, gradients = [], []
    for model in (previous, batched):
        states, centres, environment = (value.clone().requires_grad_() for value in
            (encoded.module_tokens, encoded.module_centers, encoded.env_tokens))
        case = replace(encoded, module_tokens=states, module_centers=centres, env_tokens=environment)
        state = model.prepare(case, states, phase=2, soft=soft)
        loss = sum(value.square().sum() for value in (*state.memberships.values(), *state.controls.values()))
        loss.backward()
        outcomes.append(state)
        gradients.append({"module_input": states.grad, "geometry_input": centres.grad, "environment_input": environment.grad,
                          **{name: parameter.grad for name, parameter in model.named_parameters()}})
    for tau in outcomes[0].memberships:
        torch.testing.assert_close(outcomes[0].memberships[tau], outcomes[1].memberships[tau], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(outcomes[0].controls[tau], outcomes[1].controls[tau], rtol=1e-12, atol=1e-12)
        assert outcomes[0].strategy_data["trees"][tau][0].nodes == outcomes[1].strategy_data["trees"][tau][0].nodes
    for key in gradients[0]:
        first, second = gradients[0][key], gradients[1][key]
        assert (first is None) == (second is None), key
        if first is not None:
            torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10, msg=key)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("soft", [False, True])
def test_case_batch_matches_reference_with_padding_empty_population_and_live_physics(dimension, soft):
    torch.manual_seed(804)
    encoded = EncodedInterfaceCase(torch.randn(3, 4, 8, dtype=torch.float64), torch.randn(3, 12, 8, dtype=torch.float64),
        torch.randn(3, 8, dtype=torch.float64), torch.rand(3, 4, dimension, dtype=torch.float64),
        torch.rand(3, 12, dimension, dtype=torch.float64),
        torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 1.], [0., 0., 0., 0.]], dtype=torch.float64),
        torch.randn(3, 4, 3, dtype=torch.float64), None, torch.rand(3, 12, dtype=torch.float64) + .1,
        torch.ones(dimension, dtype=torch.float64))
    batched = AdaptiveReceiverHypergraph(8).double().eval()
    with torch.no_grad():
        for head in (*batched.control_heads.values(), *batched.source_scores.values()):
            head[-1].weight.normal_(0, .1)
        batched.split_head.weight.normal_(0, .1)
    previous = deepcopy(batched)
    previous.prepare = MethodType(case_loop_prepare, previous)
    outcomes, gradients = [], []
    for candidate in (previous, batched):
        values = [value.clone().requires_grad_() for value in (encoded.module_tokens, encoded.module_centers,
            encoded.env_tokens, encoded.global_token, encoded.env_weights, encoded.coordinate_scale)]
        states, centers, environment, global_token, measure, scale = values
        case = replace(encoded, module_tokens=states, module_centers=centers, env_tokens=environment,
                       global_token=global_token, env_weights=measure, coordinate_scale=scale)
        plan = candidate.prepare(case, states, phase=2, soft=soft, capture_topology=not soft)
        sum(value.square().sum() for value in (*plan.memberships.values(), *plan.controls.values())).backward()
        outcomes.append(plan)
        gradients.append({**{str(index): value.grad for index, value in enumerate(values)},
                          **{name: parameter.grad for name, parameter in candidate.named_parameters()}})
        if not soft:
            anchor = candidate.prepare(case, states, phase=2, fixed_topology=plan)
            for tau in plan.memberships:
                torch.testing.assert_close(plan.memberships[tau], anchor.memberships[tau], rtol=0, atol=0)
                torch.testing.assert_close(plan.controls[tau], anchor.controls[tau], rtol=0, atol=0)
    for tau in outcomes[0].memberships:
        torch.testing.assert_close(outcomes[0].memberships[tau], outcomes[1].memberships[tau], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(outcomes[0].controls[tau], outcomes[1].controls[tau], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(outcomes[0].strategy_data["gates"][tau], outcomes[1].strategy_data["gates"][tau], rtol=0, atol=0)
        for first, second in zip(outcomes[0].strategy_data["trees"][tau], outcomes[1].strategy_data["trees"][tau]):
            assert first.nodes == second.nodes
    for name, first in gradients[0].items():
        second = gradients[1][name]
        assert (first is None) == (second is None), name
        if first is not None:
            torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10, msg=name)


@pytest.mark.parametrize("epoch", [3, 25, 26, 101, 301])
def test_batching_preserves_rng_curriculum_calls_and_persistent_counters(epoch):
    torch.manual_seed(1337)
    encoded = EncodedInterfaceCase(torch.randn(2, 3, 8), torch.randn(2, 9, 8), torch.randn(2, 8),
        torch.rand(2, 3, 2), torch.rand(2, 9, 2), torch.ones(2, 3), torch.randn(2, 3, 3),
        None, torch.ones(2, 9), torch.ones(2))
    batched = AdaptiveReceiverHypergraph(8).train()
    batched.set_epoch(epoch)
    batched.exercise_counter.fill_(17)
    previous = deepcopy(batched)
    previous.prepare = MethodType(case_loop_prepare, previous)
    initial_rng = torch.get_rng_state().clone()
    records = []
    for candidate in (previous, batched):
        torch.set_rng_state(initial_rng)
        calls = []
        original = candidate._exercise_depth

        def measured(original=original, calls=calls, candidate=candidate):
            calls.append(int(candidate.exercise_counter))
            return original()

        candidate._exercise_depth = measured
        plans = []
        for soft in (False, True):
            for phase in (0, 1, 2):
                plans.append(candidate.prepare(encoded, encoded.module_tokens, phase=phase, soft=soft))
        records.append((plans, torch.get_rng_state().clone(), candidate.exercise_counter.clone(), calls))
    assert len(records[0][3]) == len(records[1][3]) == 3
    assert records[0][3] == records[1][3]
    torch.testing.assert_close(records[0][1], records[1][1], rtol=0, atol=0)
    torch.testing.assert_close(records[0][2], records[1][2], rtol=0, atol=0)
    if epoch <= 25:
        torch.testing.assert_close(records[0][1], initial_rng, rtol=0, atol=0)
        assert int(records[0][2]) == 20
    for old, new in zip(records[0][0], records[1][0]):
        for tau in old.memberships:
            torch.testing.assert_close(old.strategy_data["gates"][tau], new.strategy_data["gates"][tau], rtol=0, atol=0)
