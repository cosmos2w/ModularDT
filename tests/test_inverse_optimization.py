import torch

from honf_runtime.inverse_optimization import ProposalAdam, geometry_graph, learned_graph, start_bandwidth


def test_learned_graph_measures_and_isolated_member():
    B = torch.tensor([[1., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
    graph = learned_graph(B, torch.tensor([1., 3.], dtype=torch.float64))
    expected = torch.tensor([[0., .25, 0.], [.25, 0., 0.], [0., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(graph.affinity, expected)
    torch.testing.assert_close(graph.laplacian[2], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(graph.transform[2], torch.tensor([0., 0., 1.], dtype=torch.float64))
    torch.testing.assert_close(graph.transform, graph.transform.T)


def test_geometry_graph_uses_fixed_median_scale():
    xy = torch.tensor([[0., 0.], [1., 0.], [3., 0.]], dtype=torch.float64)
    bandwidth = start_bandwidth(xy)
    torch.testing.assert_close(bandwidth, torch.tensor(2., dtype=xy.dtype))
    graph = geometry_graph(xy, bandwidth)
    torch.testing.assert_close(graph.affinity[0, 1], torch.exp(torch.tensor(-1 / 8, dtype=xy.dtype)))
    assert torch.linalg.eigvalsh(graph.laplacian).min() > -1e-12


def test_a0_matches_adam_and_guided_rms_and_refresh():
    initial = torch.tensor([[0., 0.], [1., 2.], [4., -1.]], dtype=torch.float64)
    plain = ProposalAdam(initial, .05, 'A0')
    reference = torch.nn.Parameter(initial.clone())
    adam = torch.optim.Adam([reference], lr=.05)
    project = lambda old, proposed: (proposed, {'feasible': True})
    for _ in range(3):
        loss = ((plain.xy - 2) ** 2).sum()
        diag = plain.step(loss, project)
        adam.zero_grad()
        ((reference - 2) ** 2).sum().backward()
        adam.step()
        torch.testing.assert_close(plain.xy, reference)
    assert diag['gradient_evaluations'] == 3
    B = torch.tensor([[1., .5, 0.], [0., .2, 1.]], dtype=initial.dtype)
    weights = torch.ones(2, dtype=initial.dtype)
    guided = ProposalAdam(initial, .05, 'A2')
    for epoch in range(11):
        diag = guided.step(((guided.xy - 2) ** 2).sum(), project, B, weights)
        assert diag['graph_refreshed'] == (epoch in (0, 10))
        ordinary = torch.tensor(diag['ordinary_displacement'])
        proposed = torch.tensor(diag['proposed_displacement'])
        torch.testing.assert_close(ordinary.square().mean(), proposed.square().mean())
