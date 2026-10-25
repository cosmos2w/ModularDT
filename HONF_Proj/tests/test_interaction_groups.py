"""Behavior checks for aggregation, reuse, and bounded access."""

from dataclasses import replace

import pytest
import torch
from test_contextual_joint_core import _scene

from honf_forward_core.content_hypergraph import QueryBatch
from honf_forward_core.interaction_groups import InteractionGroupCore


def scene(modules=2):
    s = _scene()
    xy = torch.cartesian_prod(torch.linspace(0.1, 1.9, 4), torch.linspace(0.1, 0.5, 2)).flip(-1)
    # fixed row-major grid for the generic fixture
    xy = torch.tensor([[x, y] for y in (0.1, 0.5) for x in (0.1, 0.7, 1.3, 1.9)])
    return replace(
        s,
        module_features=s.module_features[:, :modules],
        module_coordinates=s.module_coordinates[:, :modules],
        module_mask=s.module_mask[:, :modules],
        module_ids=s.module_ids[:, :modules],
        module_measure=s.module_measure[:, :modules],
        local_radius=s.local_radius[:, :modules],
        module_characteristic_lengths=s.module_characteristic_lengths[:, :modules],
        environment_coordinates=xy[None],
        environment_features=torch.randn(1, 8, 6),
        environment_mask=torch.ones(1, 8, dtype=torch.bool),
        environment_ids=torch.arange(8)[None],
        environment_measure=torch.ones(1, 8),
        environment_characteristic_lengths=torch.ones(1, 8, 2) * 0.1,
    )


def core(arm="G-F"):
    return InteractionGroupCore(
        architecture=arm, context_dim=16, local_hidden_dim=16, environment_nx=4, environment_ny=2, query_tile_size=3
    ).eval()


def query(n=19):
    return QueryBatch(
        torch.rand(1, n, 2) * torch.tensor([2.0, 0.6]),
        scene_key="contextual-scene",
        coordinate_frame_key="physical-frame",
        query_ids=torch.arange(n)[None],
    )


@pytest.mark.parametrize("arm", ["G-F", "G-EE", "G-H"])
@pytest.mark.parametrize("modules", [0, 1, 2])
def test_native_shape_gradient_caps_and_reload(arm, modules):
    c = core(arm)
    s = scene(modules)
    q = query()
    p = c.prepare(s)
    out, _tr = c.decode(p, q, return_trace=True)
    assert out.joint_fields.shape == (1, 19, 4)
    assert torch.isfinite(out.joint_fields).all()
    assert out.work_counts["original_e_query_reads"] == 0
    assert out.work_counts["module_value_query_reads"] == 0
    assert out.work_counts["group_value_reads"] <= 19 * 8
    if arm != "G-H":
        assert p.work_counts["policy_rows"] == p.work_counts["metadata_nodes"] == 0
    if arm != "G-EE":
        assert p.work_counts["ee_pairs"] == 0
    out.joint_fields.square().sum().backward()
    assert c.layers[-1].gm.v.weight.grad is not None
    if modules:
        assert c.proposal_key.weight.grad.abs().sum() > 0
        assert c.layers[-1].gm.out.weight.grad.abs().sum() > 0
    restored = InteractionGroupCore.from_config(c.get_config()).eval()
    restored.load_state_dict(c.state_dict())
    torch.testing.assert_close(out.joint_fields, restored(s, q).joint_fields)
    # Chunking is the same restricted function, including TRAIN exploration.
    old = c.query_tile_size
    c.query_tile_size = 128
    torch.testing.assert_close(out.joint_fields, c.decode(c.prepare(s), q).joint_fields, atol=2e-6, rtol=2e-6)
    c.query_tile_size = old


def test_shared_key_value_and_early_final_group_dependence():
    c = core()
    s = scene(1)
    q = query()
    p = c.prepare(s)
    g = p.final_groups[0]
    k, v = c.layers[-1].gm.project(g.states)
    torch.testing.assert_close(k, g.keys)
    torch.testing.assert_close(v, g.values)
    ordinary = c.decode(p, q).joint_fields
    for layer in (0, 1):
        changed = c.decode(c.prepare(s, clamp_layer=layer), q).joint_fields
        assert (ordinary - changed).abs().max() > 1e-7
    with torch.no_grad():
        c.field_head.weight.add_(0.01)
    with pytest.raises(ValueError):
        c.decode(p, q)


def test_frontier_reversible_complete_local_and_sample_replay():
    c = core("G-H").train()
    s = scene()
    query()
    a = c.prepare(s)
    b = c.prepare(s)
    for ga, gb in zip(a.layers, b.layers):
        torch.testing.assert_close(ga[0].labels, gb[0].labels)
        assert ga[0].frontier.edge_count == 10
        assert len(ga[0].frontier.children) == 15
    for bias, k in ((10.0, 1), (-10.0, 8), (10.0, 1)):
        with torch.no_grad():
            for layer in c.layers:
                layer.frontier.policy[-1].weight.zero_()
                layer.frontier.policy[-1].bias.fill_(bias)
        c.eval()
        p = c.prepare(s)
        assert all(layer[0].frontier.K == k for layer in p.layers)
        for layer in p.layers:
            assert torch.equal(torch.unique(layer[0].labels), torch.arange(k))


def test_zero_information_mass_is_null_and_common_initialization_matches():
    c = core()
    s = scene(1)
    s = replace(s, environment_measure=torch.zeros_like(s.environment_measure))
    p = c.prepare(s)
    g = p.final_groups[0]
    assert torch.count_nonzero(g.values) == 0 or torch.count_nonzero(g.B) == 0
    assert torch.count_nonzero(g.B) == 0
    assert torch.isfinite(c.decode(p, query()).joint_fields).all()
    a, b = core("G-F").state_dict(), core("G-EE").state_dict()
    for name, value in a.items():
        torch.testing.assert_close(value, b[name], rtol=0, atol=0)


def test_tiny_tree_expected_task_work_score_function():
    from honf_forward_core.interaction_groups.frontier import LocalFrontier

    policy = LocalFrontier(4, 2, 1)
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.1, 0.3]])
    r = torch.ones(1, 2, 1) * 0.3
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    mass = torch.ones(2)
    frontier = policy(x, r, xy, mass, adaptive=True, training=False, epoch=1, seed=(0,))
    p = frontier.probabilities[0]
    losses = x.new_tensor([0.7, 1.2])
    costs = x.new_tensor([1.0, 2.0])
    price = 0.03
    objective = losses + price * costs
    expectation = p * objective[0] + (1 - p) * objective[1]
    actual = torch.autograd.grad(expectation, policy.policy[-1].bias, retain_graph=True)[0]
    # Sum of every legal sampled outcome's score-function contribution, with
    # an action-independent arbitrary baseline.
    score = p.detach() * (objective[0] - 0.8) * p.log() + (1 - p.detach()) * (objective[1] - 0.8) * (1 - p).log()
    enumerated = torch.autograd.grad(score, policy.policy[-1].bias)[0]
    torch.testing.assert_close(actual, enumerated, rtol=1e-5, atol=1e-7)


def test_mass_additive_identity_and_linear_grid_edge_count():
    from honf_forward_core.interaction_groups.frontier import grid_edges
    from honf_forward_core.interaction_groups.model import Exchange

    layer = Exchange(16, 4)
    receiver = torch.randn(1, 16)
    state = torch.randn(1, 16)
    keys, values = layer.project(state)
    a, _ = layer.read(receiver, keys, values, torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([0.7]))
    b, _ = layer.read(
        receiver,
        keys.expand(2, -1, -1),
        values.expand(2, -1, -1),
        torch.zeros(1, 2),
        torch.zeros(2, 2),
        torch.tensor([0.2, 0.5]),
    )
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    assert len(grid_edges(24, 8)) == 352
    assert len(grid_edges(48, 8)) < 2.1 * len(grid_edges(24, 8))


def test_training_exploration_never_opens_padded_module():
    c = core().train()
    s = scene(3)
    s = replace(s, module_mask=torch.tensor([[True, False, True]]))
    out, trace = c.decode(c.prepare(s), query(48), return_trace=True)
    assert out.work_counts["exploration_queries"] == 3
    for tile in trace[0]:
        assert (tile["proposal_ids"] != 1).all()


def test_streamed_task_policy_gradients_baseline_and_retry_state():
    from types import SimpleNamespace

    from channelthermal.training.interaction_group_objective import InteractionGroupObjectiveRuntime

    adapter = torch.nn.Module()
    adapter.core = core("G-H").train()
    adapter.architecture = "G-H"
    p = adapter.core.prepare(scene())
    out = adapter.core.decode(p, query())
    loss = out.joint_fields.square().mean((1, 2))
    runtime = InteractionGroupObjectiveRuntime(adapter, SimpleNamespace(recipe={"objective": {"work_prices": {}}}))
    prediction = SimpleNamespace(native_prepared=SimpleNamespace(organization=p, query_batch=query()))
    runtime.collect(prediction, loss, 1)
    state = runtime.state_dict()
    assert runtime.state["task_baseline"] == 0
    loss.mean().backward()
    runtime.before_optimizer_step(adapter, None, {}, {}, epoch=1, update_index=0)
    assert runtime.state["task_baseline"] > 0
    assert runtime.state["updates"][-1]["native_policy_norm"] > 0
    assert adapter.core.layers[0].frontier.policy[-1].weight.grad.abs().sum() > 0
    runtime.load_state_dict(state)
    assert runtime.state["task_baseline"] == 0 and len(runtime.pending) == 1


@pytest.mark.parametrize("prior", ["legacy_additive", "balanced_postings_v1"])
def test_corrected_exhaustive_prior_and_separate_coverage(prior):
    c = core()
    c.reader_prior = prior
    p = c.prepare(scene())
    q = query(7)
    restricted, tr = c.decode(p, q, return_trace=True, association_scope="all")
    full, tf = c.decode(p, q, return_trace=True, exhaustive=True)
    for rt, ft in zip(tr[0], tf[0]):
        candidates = rt["candidates"]
        valid = candidates >= 0
        # The same group's B contribution is identical with either support;
        # exhaustive must not add an existing posting contribution twice.
        actual = ft["association_prior"].gather(1, candidates.clamp_min(0))
        torch.testing.assert_close(rt["association_prior"][valid], actual[valid])
    if prior == "balanced_postings_v1":
        for t in tf[0]:
            torch.testing.assert_close(t["association_prior"].sum(-1), t["alpha"].sum(-1))
    assert full.work_counts["group_value_reads"] == 7 * 8
    assert restricted.work_counts["original_e_query_reads"] == 0


def test_balanced_reader_normalizes_each_entrance_and_retains_module_gradient():
    c = core()
    c.reader_prior = "balanced_postings_v1"
    p = c.prepare(scene(1))
    out, tr = c.decode(p, query(7), return_trace=True)
    for t in tr[0]:
        torch.testing.assert_close(t["geometry_prior"].sum(-1), torch.ones(len(t["candidates"])))
        torch.testing.assert_close(t["association_prior"].sum(-1), t["alpha"].sum(-1), atol=1e-6, rtol=1e-6)
    out.joint_fields.square().sum().backward()
    assert c.proposal_key.weight.grad.abs().sum() > 0


def test_retained_ee_adaptive_pools_instead_of_member_expansion():
    c = core("G-EE-H")
    s = scene()
    p = c.prepare(s, forced_frontier={0: (14, "parent"), 1: (14, "parent")})
    assert [layer[0].frontier.K for layer in p.layers] == [1, 1]
    assert p.work_counts["ee_pairs"] > 0
    out = c.decode(p, query(7))
    assert out.work_counts["group_value_reads"] == 7
    assert out.work_counts["original_e_query_reads"] == 0
    assert p.work_counts["group_key_value_rows"] == 2
    restored = InteractionGroupCore.from_config(c.get_config()).eval()
    restored.load_state_dict(c.state_dict())
    assert restored.adaptive_groups and restored.layers[0].ee is not None


def test_legacy_serialized_config_does_not_gain_default_recipe_fields():
    c = core("G-EE")
    config = c.get_config()
    assert "reader_prior" not in config and "frontier_distance_scales" not in config
    restored = InteractionGroupCore.from_config(config)
    assert restored.get_config() == config
    restored.load_state_dict(c.state_dict(), strict=True)
