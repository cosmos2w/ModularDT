"""Batched receiver access preserves scalar tree math and prepared-state ownership."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.interface_fields.adaptive_interaction_cover import CaseLocalReceiverTree, ReceiverAnchorUniverse
from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph
from honf_forward_core.interface_fields.receiver_tree_access import build_receiver_tree_geometry, receiver_tree_access
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("endpoints", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
def test_batched_access_scalar_reference_and_all_first_gradients(dimension, dtype, endpoints, depth):
    torch.manual_seed(1804)
    capacity = 2 ** (depth + 1) - 1
    initial_coords = [torch.rand(count, dimension, dtype=dtype) for count in (3, 27)]
    initial_query = torch.rand(2, 23, dimension, dtype=dtype)
    initial_gate = torch.rand(2, capacity, dtype=dtype)
    if endpoints:
        initial_gate = (initial_gate > .5).to(dtype)
    outputs, gradients = [], []
    for optimized in (False, True):
        coordinates = [value.clone().requires_grad_() for value in initial_coords]
        scales = [torch.ones(dimension, dtype=dtype, requires_grad=True) for _ in coordinates]
        query, gates = initial_query.clone().requires_grad_(), initial_gate.clone().requires_grad_()
        trees = [CaseLocalReceiverTree.build(ReceiverAnchorUniverse(value, torch.ones(value.shape[0], dtype=dtype),
            torch.zeros(value.shape[0], dtype=torch.long), scale), max_nodes=capacity, min_leaf_anchors=1, max_depth=depth)
            for value, scale in zip(coordinates, scales)]
        if optimized:
            output = receiver_tree_access(query, gates, build_receiver_tree_geometry(trees, capacity), depth)
        else:
            output = torch.stack([F.pad(tree.access(query[case], gates[case, :len(tree.nodes)]),
                (0, capacity-len(tree.nodes))) for case, tree in enumerate(trees)])
        output.square().mean().backward()
        outputs.append(output)
        gradients.append([query.grad, gates.grad, *[value.grad for value in coordinates], *[value.grad for value in scales]])
    rtol, atol = (2e-4, 2e-5) if dtype == torch.float32 else (1e-12, 1e-12)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=rtol, atol=atol)
    for first, second in zip(*gradients):
        torch.testing.assert_close(first, second, rtol=rtol, atol=atol)


def test_access_rejects_invalid_real_gates_and_geometry_axes():
    coords = torch.tensor([[0., 0.], [1., 1.]])
    tree = CaseLocalReceiverTree.build(ReceiverAnchorUniverse(coords, torch.ones(2), torch.zeros(2, dtype=torch.long), torch.ones(2)),
                                       max_nodes=3, min_leaf_anchors=1, max_depth=1)
    geometry = build_receiver_tree_geometry([tree], 3)
    queries, gates = coords[None], torch.zeros(1, 3)
    for value in (-.1, 1.1, torch.nan):
        invalid = gates.clone()
        invalid[0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            receiver_tree_access(queries, invalid, geometry, 1)
    with pytest.raises(ValueError, match="axes"):
        receiver_tree_access(torch.zeros(1, 2, 3), gates, geometry, 1)
    empty = receiver_tree_access(queries[:, :0], gates, geometry, 1)
    assert empty.shape == (1, 0, 3)


@pytest.mark.parametrize("dimension", [2, 3])
def test_later_phase_prepares_do_not_mutate_old_geometry_controls_or_gradient_graph(dimension):
    torch.manual_seed(144)
    encoded = EncodedInterfaceCase(torch.randn(2, 4, 8, dtype=torch.float64), torch.randn(2, 12, 8, dtype=torch.float64),
        torch.randn(2, 8, dtype=torch.float64), torch.rand(2, 4, dimension, dtype=torch.float64),
        torch.rand(2, 12, dimension, dtype=torch.float64), torch.ones(2, 4, dtype=torch.float64),
        torch.randn(2, 4, 3, dtype=torch.float64), None, torch.rand(2, 12, dtype=torch.float64) + .1,
        torch.ones(dimension, dtype=torch.float64))
    organizer = AdaptiveReceiverHypergraph(8).double().eval()
    with torch.no_grad():
        for head in (*organizer.control_heads.values(), *organizer.source_scores.values()):
            head[-1].weight.normal_(0, .1)
    old_coords = encoded.env_coords.clone().requires_grad_()
    first = replace(encoded, env_coords=old_coords, env_tokens=encoded.env_tokens.clone().requires_grad_())
    previous = organizer.prepare(first, first.module_tokens, phase=0)
    queries = torch.rand(2, 11, dimension, dtype=torch.float64)
    snapshot = organizer.access(previous, queries, "QE").control.detach().clone()
    new_coords = (encoded.env_coords + .2).requires_grad_()
    second = replace(encoded, module_centers=encoded.module_centers + .4, env_coords=new_coords,
                     env_tokens=encoded.env_tokens + 1, global_token=encoded.global_token + 2)
    refreshed = organizer.prepare(second, second.module_tokens + .5, phase=2)
    after = organizer.access(previous, queries, "QE")
    torch.testing.assert_close(snapshot, after.control.detach(), rtol=0, atol=0)
    assert not torch.allclose(previous.controls["QE"], refreshed.controls["QE"])
    assert not torch.allclose(previous.strategy_data["access_geometry"]["Q"]["boundary"],
                              refreshed.strategy_data["access_geometry"]["Q"]["boundary"])
    old_gradient, new_gradient = torch.autograd.grad(after.control.square().sum(), (old_coords, new_coords), allow_unused=True)
    assert old_gradient is not None and old_gradient.norm() > 0
    assert new_gradient is None
    order = torch.arange(10, -1, -1)
    ordered = organizer.access(previous, queries[:, order], "QE")
    parts = [organizer.access(previous, queries[:, start:start+3], "QE") for start in range(0, 11, 3)]
    for key in ("density", "weight", "control", "edge_access"):
        value = getattr(after, key)
        torch.testing.assert_close(value, getattr(ordered, key)[:, order], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(value, torch.cat([getattr(part, key) for part in parts], 1), rtol=1e-12, atol=1e-12)
