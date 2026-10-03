"""Task-trained receiver frontier, typed validity and reusable state checks."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _case(dimension=2, modules=4, environment=12):
    torch.manual_seed(61)
    hidden = 8
    coords = torch.rand(1, modules, dimension)
    env = torch.rand(1, environment, dimension)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(1, modules, hidden), env_tokens=torch.randn(1, environment, hidden),
        global_token=torch.randn(1, hidden), module_centers=coords, env_coords=env,
        module_present=torch.ones(1, modules), module_features=torch.randn(1, modules, 3),
        env_features=None, env_weights=torch.ones(1, environment), coordinate_scale=torch.ones(dimension),
        receiver_anchor_coords=env.clone(), receiver_anchor_weights=torch.ones(1, environment),
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_identical_parent_children_keep_full_fine_operator(dimension):
    encoded = _case(dimension)
    organizer = AdaptiveReceiverHypergraph(8, spatial_dim=dimension)
    query = torch.rand(1, 7, dimension)
    for _ in range(4):  # scheduled root, depth1, depth2 and depth3 exercises
        state = organizer.prepare(encoded, encoded.module_tokens)
        for tau, receiver in (("MM", encoded.module_centers), ("ME", encoded.module_centers),
                              ("EM", encoded.env_coords), ("QM", query), ("QE", query)):
            access = organizer.access(state, receiver, tau)
            expected = torch.ones_like(access.weight)
            if tau == "MM":
                expected[:, torch.arange(4), torch.arange(4)] = 0
            torch.testing.assert_close(access.weight, expected)
            initial_control = torch.linspace(-0.1, 0.1, organizer.control_dim).expand_as(access.control)
            initial_control = torch.where(access.support[..., None], initial_control, torch.zeros_like(initial_control))
            torch.testing.assert_close(access.control, initial_control)
            torch.testing.assert_close(access.edge_access.sum(-1), torch.ones_like(access.weight[..., 0]))


def test_real_optimizer_learns_nonidentical_action_and_split_gradient():
    encoded = _case()
    organizer = AdaptiveReceiverHypergraph(8, organizer_dim=16, control_dim=2).eval()
    optimizer = torch.optim.Adam(organizer.parameters(), lr=0.04)
    query = encoded.receiver_anchor_coords
    source_values = encoded.env_coords[..., 0]
    target = query[..., 0]
    learned_gradient = 0.0
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        state = organizer.prepare(encoded, encoded.module_tokens, soft=True)
        access = organizer.access(state, query, "QE")
        prediction = (access.weight * source_values[:, None]).mean(-1) + access.control[..., 0].mean(-1)
        loss = (prediction - target).square().mean()
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in organizer.parameters())
        learned_gradient = max(learned_gradient, float(organizer.split_head.weight.grad.abs().sum()))
        optimizer.step()
    assert learned_gradient > 1e-6
    hard = organizer.prepare(encoded, encoded.module_tokens)
    split = {tau: gate.clone() for tau, gate in hard.strategy_data["gates"].items()}
    root = replace(hard, strategy_data={**hard.strategy_data, "gates": {tau: torch.zeros_like(gate) for tau, gate in split.items()}})
    leaves = replace(hard, strategy_data={**hard.strategy_data, "gates": {tau: torch.ones_like(gate) for tau, gate in split.items()}})
    parent = organizer.access(root, query, "QE")
    children = organizer.access(leaves, query, "QE")
    assert max(float((parent.weight - children.weight).abs().max()),
               float((parent.control - children.control).abs().max())) > 1e-4


def test_query_permutation_chunks_and_phase_freshness():
    encoded = _case()
    organizer = AdaptiveReceiverHypergraph(8).eval()
    with torch.no_grad():
        organizer.geometry_strength["QE"].fill_(0.8)
        organizer.control_heads["QE"][-1].weight.normal_(0, 0.1)
    state = organizer.prepare(encoded, encoded.module_tokens, phase=0)
    query = torch.rand(1, 9, 2)
    complete = organizer.access(state, query, "QE")
    order = torch.randperm(9)
    shuffled = organizer.access(state, query[:, order], "QE")
    torch.testing.assert_close(shuffled.weight, complete.weight[:, order])
    chunks = [organizer.access(state, query[:, start:start+3], "QE").weight for start in range(0, 9, 3)]
    torch.testing.assert_close(torch.cat(chunks, 1), complete.weight)
    next_phase = organizer.prepare(encoded, encoded.module_tokens + 0.4, phase=1)
    assert next_phase.phase == 1
    assert not torch.allclose(state.controls["QE"], next_phase.controls["QE"])


@pytest.mark.parametrize("modules,environment", [(0, 0), (1, 0), (0, 3), (1, 3)])
def test_empty_types_and_no_mm_pairs_are_finite(modules, environment):
    encoded = _case(modules=modules, environment=environment)
    organizer = AdaptiveReceiverHypergraph(8).eval()
    for soft in (False, True):
        state = organizer.prepare(encoded, encoded.module_tokens, soft=soft)
        query = torch.rand(1, 3, 2)
        for tau, receiver in (("MM", encoded.module_centers), ("ME", encoded.module_centers),
                              ("EM", encoded.env_coords), ("QM", query), ("QE", query)):
            access = organizer.access(state, receiver, tau)
            assert torch.isfinite(access.weight).all()
            assert torch.isfinite(access.control).all()
        if modules <= 1:
            assert not organizer.access(state, encoded.module_centers, "MM").support.any()


def test_padding_and_module_permutation_preserve_native_rows():
    encoded = _case()
    encoded = replace(encoded, module_present=torch.tensor([[1., 1., 1., 0.]]))
    organizer = AdaptiveReceiverHypergraph(8).eval()
    with torch.no_grad():
        organizer.geometry_strength["MM"].fill_(2)
        organizer.source_scores["MM"][-1].weight.normal_(0, 0.1)
    state = organizer.prepare(encoded, encoded.module_tokens)
    expected = organizer.access(state, encoded.module_centers, "MM")
    order = torch.tensor([2, 0, 3, 1])
    permuted = replace(encoded, module_tokens=encoded.module_tokens[:, order],
                       module_centers=encoded.module_centers[:, order], module_present=encoded.module_present[:, order],
                       module_features=encoded.module_features[:, order])
    actual = organizer.access(organizer.prepare(permuted, permuted.module_tokens), permuted.module_centers, "MM")
    torch.testing.assert_close(actual.weight, expected.weight[:, order][:, :, order])
    assert not expected.support[:, 3].any() and not expected.support[:, :, 3].any()
    damaged = replace(encoded, module_tokens=encoded.module_tokens.clone(), module_centers=encoded.module_centers.clone())
    damaged.module_tokens[:, 3] = torch.nan
    damaged.module_centers[:, 3] = torch.nan
    padded = organizer.access(organizer.prepare(damaged, damaged.module_tokens), damaged.module_centers, "MM")
    torch.testing.assert_close(padded.weight, expected.weight)


def test_soft_restores_hard_omission_without_encoder_gradient_and_resumes_curriculum():
    encoded = _case()
    states = encoded.module_tokens.clone().requires_grad_()
    organizer = AdaptiveReceiverHypergraph(8).train()
    with torch.no_grad():
        organizer.geometry_strength["QE"].fill_(8)
    query = torch.tensor([[[0.05, 0.05]]])
    hard = organizer.access(organizer.prepare(encoded, states), query, "QE")
    omitted = ~hard.support
    assert omitted.any()
    soft_state = organizer.prepare(encoded, states, soft=True)
    soft = organizer.access(soft_state, query, "QE")
    assert (soft.weight[omitted] > 0).all()
    soft.weight[omitted].sum().backward()
    assert organizer.geometry_strength["QE"].grad.abs() > 0
    assert states.grad is None
    organizer.train().set_epoch(301)
    organizer.exercise_counter.fill_(17)
    reloaded = AdaptiveReceiverHypergraph(8)
    reloaded.load_state_dict(organizer.state_dict())
    assert int(reloaded.training_epoch) == 301 and int(reloaded.exercise_counter) == 17
    reloaded.eval()
    assert reloaded.prepare(encoded, states).diagnostics["exploration"].item() == 0


def test_frozen_eval_control_moment_keeps_caller_owned_heat_gradient():
    encoded = _case()
    organizer = AdaptiveReceiverHypergraph(8).double().eval().requires_grad_(False)
    with torch.no_grad():
        organizer.control_heads["QM"][-1].weight.normal_(0, .2)
    heat = torch.tensor([[.3, .5, .7, .9]], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([[1., -1., .5, -.5]], dtype=heat.dtype)
    basis = encoded.module_tokens.double()
    encoded = replace(encoded, module_tokens=basis, env_tokens=encoded.env_tokens.double(),
        global_token=encoded.global_token.double(), module_centers=encoded.module_centers.double(),
        env_coords=encoded.env_coords.double(), module_present=encoded.module_present.double(),
        env_weights=encoded.env_weights.double(), coordinate_scale=encoded.coordinate_scale.double(),
        receiver_anchor_coords=encoded.receiver_anchor_coords.double(),
        receiver_anchor_weights=encoded.receiver_anchor_weights.double())

    def moment(value):
        states = basis * value[..., None]
        case = replace(encoded, module_tokens=states, global_token=states.mean(1))
        plan = organizer.prepare(case, states)
        return organizer.access(plan, encoded.receiver_anchor_coords, "QM").control[..., 0].sum()

    gradient, = torch.autograd.grad(moment(heat), heat)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 1e-6
    epsilon = 1e-5
    finite_difference = (moment(heat.detach() + epsilon * direction) -
                         moment(heat.detach() - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), finite_difference, rtol=1e-4, atol=1e-7)
    assert heat.grad is None and all(parameter.grad is None for parameter in organizer.parameters())


def test_quadrature_atom_split_preserves_query_action_and_collective_control():
    encoded = _case()
    organizer = AdaptiveReceiverHypergraph(8).eval()
    with torch.no_grad():
        organizer.geometry_strength["QE"].fill_(0.8)
        organizer.source_scores["QE"][-1].weight.normal_(0, 0.2)
        organizer.control_heads["QE"][-1].weight.normal_(0, 0.1)
    # Split one environmental source atom, keeping the receiver index fixed.
    weights = encoded.env_weights.clone()
    weights[:, 0] *= 0.5
    split = replace(encoded, env_coords=torch.cat((encoded.env_coords, encoded.env_coords[:, :1]), 1),
                    env_tokens=torch.cat((encoded.env_tokens, encoded.env_tokens[:, :1]), 1),
                    env_weights=torch.cat((weights, weights[:, :1]), 1))
    query = torch.rand(1, 5, 2)
    for soft in (False, True):
        base_state = organizer.prepare(encoded, encoded.module_tokens, soft=soft)
        split_state = organizer.prepare(split, split.module_tokens, soft=soft)
        base = organizer.access(base_state, query, "QE")
        actual = organizer.access(split_state, query, "QE")
        torch.testing.assert_close(actual.weight[..., :-1], base.weight)
        torch.testing.assert_close(actual.control[..., :-1, :], base.control)
        torch.testing.assert_close(actual.weight[..., -1], base.weight[..., 0])
        torch.testing.assert_close(split_state.controls["QE"], base_state.controls["QE"])


def test_receiver_catalogue_roles_and_index_reuse_are_phase_local():
    encoded = _case()
    encoded = replace(encoded, receiver_anchor_roles=torch.arange(12)[None] % 3)
    organizer = AdaptiveReceiverHypergraph(8).eval()
    state = organizer.prepare(encoded, encoded.module_tokens)
    trees = state.strategy_data["trees"]
    assert trees["MM"][0] is trees["ME"][0]
    assert trees["QM"][0] is trees["QE"][0]
    assert trees["MM"][0] is not trees["QE"][0]
    torch.testing.assert_close(trees["QE"][0].universe.roles, encoded.receiver_anchor_roles[0])
    refreshed = organizer.prepare(encoded, encoded.module_tokens, phase=1)
    assert refreshed.strategy_data["trees"]["QM"][0] is not trees["QM"][0]
