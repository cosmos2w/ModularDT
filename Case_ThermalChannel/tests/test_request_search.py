import numpy as np
import pytest

from channelthermal.inverse.request_search import (
    ATTEMPTS, SolveBudget, block_probabilities, cross_swap, geometry_scores,
    physical_accept, propose, pullback_weights, roi_mask, routing_scores, select_predicted,
)


def test_stencil_pullback_and_global_trace_mapping():
    w = pullback_weights([[3, 1], [1, 2]], [[.25, .75], [.5, .5]], [1, 1], [.4, .6], 4)
    np.testing.assert_allclose(w, [0, .6, .3, .1])
    D = np.array([[1., .5], [0., 1.]])
    beta = np.array([[.2, .8], [.7, .3], [.4, .6]])
    lam = np.array([[1., 0.], [.5, .5], [.25, .75]])
    s, h, a = routing_scores(w, [(np.array([3, 1]), beta[:2], lam[:2]), ([2], beta[2:], lam[2:])], D)
    np.testing.assert_allclose(s, .1 * beta[0] + .6 * beta[1] + .3 * beta[2])
    np.testing.assert_allclose(h, sum(a[i] * np.outer(D[i], D[i]) for i in range(2)))
    with pytest.raises(ValueError, match='missing'):
        routing_scores(w, [([3], beta[:1], lam[:1])], D)


def test_common_attempts_controls_and_block_support():
    ids = [0, 1, 2]
    h = block_probabilities(ids, [1, 0, 0], np.zeros((3, 3)))
    g, a = geometry_scores([[2, 2], [5, 2], [8, 2]], .5, [10, 5], 'A')
    g = block_probabilities(ids, g, a)
    assert np.all(h[2][1] > 0) and len(h[2][0]) == 3
    source = {'module_centers': np.array([[2., 2.], [5., 2.], [8., 2.]], dtype=np.float64), 'heat': [1, 2, 3]}
    for attempt in range(ATTEMPTS):
        x = propose(source, ids, h, .5, 0, 0, attempt, lambda _: False)
        y = propose(source, ids, g, .5, 0, 0, attempt, lambda _: True)
        assert not x['valid'] and x['attempt'] == attempt
        assert x['structure']['heat'] == source['heat']
        for i in set(x['block']) & set(y['block']):
            np.testing.assert_array_equal(x['structure']['module_centers'][i], y['structure']['module_centers'][i])
        assert np.linalg.norm(x['structure']['module_centers'] - source['module_centers'], axis=1).max() <= .0750000001
    assert np.array_equal(source['module_centers'], [[2, 2], [5, 2], [8, 2]])


def test_selection_acceptance_and_budget():
    rows = [{'valid': True, 'f': 7., 'delta_p': 2.}, {'valid': True, 'f': 9., 'delta_p': 0.}, {'valid': True, 'f': 9., 'delta_p': 0.}]
    assert select_predicted(10, rows, 1) is rows[1]
    assert select_predicted(8, rows, 1) is None
    assert physical_accept(10, 9, 1.1, 1) == 'baseline'
    assert physical_accept(10, 9, 0, 1, solved=False) == 'unresolved'
    assert cross_swap(11, 9, 12, 8, 2, 4) == 1
    b = SolveBudget(max_starts=1)
    assert b.start('x') and not b.start('x')
    b.finish('x', {'success': False}, .2)
    with pytest.raises(RuntimeError): b.start('y')


def test_roi_uses_expanded_baseline_disks():
    p = np.array([[3., 2.], [3.5, 2.], [3., 3.5]])
    mask = roi_mask(p, [10, 10], [[3, 2]], [1], .4, 'A')
    assert mask.tolist() == [False, True, True]


def test_tiled_capture_uses_anchored_directory_and_global_indices():
    import torch
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    from honf_forward_core.content_hypergraph.contracts import QueryBatch
    from channelthermal.inverse.request_search import capture_routing_scores

    class Directory:
        def row(self, batch, anchor):
            return [NS(member_index=torch.tensor([0, 2]), access_weight=torch.tensor([.3, .7])),
                    NS(member_index=torch.tensor([1, 3]), access_weight=torch.tensor([.4, .6]))][anchor]

    class Core:
        query_tile_size = 2
        def read_execution(self, mode):
            assert mode == 'restricted'
            return nullcontext()
        def capture_interaction_routing_trace(self, organization, q):
            assert q.num_queries <= 2 and q.anchor_indices is not None
            ids = q.query_ids[0].to(torch.float64)
            return {'member_weights_M': torch.stack((ids / 10, ids / 20), -1)[None],
                    'anchor_indices': q.anchor_indices, 'anchor_weights': q.anchor_weights}

    query = QueryBatch(coordinates=torch.zeros(1, 5, 2), query_ids=torch.arange(5)[None],
        anchor_indices=torch.tensor([[[0], [1], [0], [1], [0]]]), anchor_weights=torch.ones(1, 5, 1))
    scene = NS(module_ids=torch.tensor([[10, 11]]), environment_ids=torch.tensor([[20, 21]]))
    prepared = NS(query_batch=query, organization=NS(scene=scene, e_access_directory=NS(restricted=Directory())))
    got = capture_routing_scores(NS(core=Core()), None, prepared, [0, .25, 0, 0, .75])
    np.testing.assert_allclose(got['service'], [.325, .1625])
    np.testing.assert_allclose(got['joint_membership'], [[.3, 0, .7, 0], [0, .4, 0, .6]], rtol=1e-6)
    np.testing.assert_allclose(got['address_mass'], [.75, .25])
    assert got['query_indices'].tolist() == [1, 4]
