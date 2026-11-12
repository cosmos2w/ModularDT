import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph.staged_attention_v2 import ContextualPairMessage, SpatialRouteAttentionV2
from honf_forward_core.continuous_joint.model import ContinuousJointCore


def test_tiled_pair_matches_direct_values_and_gradients():
    torch.manual_seed(3)
    module = ContextualPairMessage(128, 4, 5).double()
    with torch.no_grad():
        module.output.normal_()
    receiver = torch.randn(2, 3, 128, dtype=torch.double, requires_grad=True)
    donor = torch.randn(2, 4, 128, dtype=torch.double, requires_grad=True)
    geometry = torch.randn(2, 3, 4, 5, dtype=torch.double, requires_grad=True)
    coefficients = torch.rand(2, 3, 4, 4, dtype=torch.double, requires_grad=True)
    eligible = torch.rand(2, 3, 4) > .4
    tiled = module(receiver, donor, geometry, coefficients, eligible, tile_size=3)
    a = torch.einsum('brc,hcp->brhp', receiver, module.receiver)[:, :, None] + torch.einsum('brdg,hgp->brdhp', geometry, module.geometry) + module.bias
    q = torch.einsum('bdc,hcp->bdhp', donor, module.donor)[:, None]
    psi = F.silu(a + q) - F.silu(a)
    reduced = (psi * coefficients[..., None] * eligible[..., None, None]).sum(2)
    direct = torch.einsum('brhp,hpv->brhv', reduced, module.output).reshape(2, 3, 128)
    args = (receiver, donor, geometry, coefficients, *module.parameters())
    grads_tiled = torch.autograd.grad(tiled.square().sum(), args, retain_graph=True)
    grads_direct = torch.autograd.grad(direct.square().sum(), args)
    torch.testing.assert_close(tiled, direct)
    for actual, expected in zip(grads_tiled, grads_direct):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


def test_zero_initialization_and_route_learning_and_masks():
    torch.manual_seed(7)
    plain = SpatialRouteAttentionV2(128, 4, 2)
    augmented = SpatialRouteAttentionV2(128, 4, 2, contextual_message=True)
    augmented.load_state_dict(plain.state_dict(), strict=False)
    states = torch.randn(1, 3, 128, requires_grad=True)
    coords = torch.randn(1, 3, 2)
    lengths = torch.ones(1, 3, 2)
    measures = torch.ones(1, 3)
    mask = torch.tensor([[True, True, False]])
    args = (states, states, coords, coords, lengths, lengths, measures, torch.ones(1), mask, mask, torch.ones(1, 2))
    original = plain(*args, remove_self=True)[0]
    out = augmented(*args, remove_self=True)[0]
    torch.testing.assert_close(out, original)
    base_grads = torch.autograd.grad(original.square().sum(), tuple(plain.parameters()), retain_graph=True)
    inherited = dict(augmented.named_parameters())
    new_grads = torch.autograd.grad(out.square().sum(), tuple(inherited[n] for n, _ in plain.named_parameters()), retain_graph=True)
    for left, right in zip(base_grads, new_grads):
        torch.testing.assert_close(left, right)
    out.square().sum().backward()
    assert augmented.pair_message.output.grad.abs().sum() > 0
    with torch.no_grad():
        augmented.pair_message.output.normal_(std=.1)
    augmented.zero_grad()
    out = augmented(*args, remove_self=True)[0]
    assert not torch.allclose(out, original)
    assert out[:, 2].count_nonzero() == 0
    out.square().sum().backward()
    for p in (augmented.pair_message.receiver, augmented.pair_message.donor, augmented.pair_message.geometry):
        assert p.grad.abs().sum() > 0
    perm = torch.tensor([1, 0, 2])
    pargs = (states[:, perm], states[:, perm], coords[:, perm], coords[:, perm], lengths[:, perm], lengths[:, perm], measures[:, perm], torch.ones(1), mask[:, perm], mask[:, perm], torch.ones(1, 2))
    torch.testing.assert_close(augmented(*pargs, remove_self=True)[0], out[:, perm], atol=1e-5, rtol=1e-5)


def test_config_source_inventory_and_rng():
    base = dict(schema='content_hypergraph_core_v2', architecture='A2-D', module_input_dim=3, environment_input_dim=3, coordinate_dim=2, thermal_response_rank=0)
    # Use the declared schema rather than relying on an old checkpoint spelling.
    from honf_forward_core.content_hypergraph.model_v2 import ContentHypergraphCoreV2
    base['schema'] = ContentHypergraphCoreV2.schema
    source = ContinuousJointCore(base)
    rng = torch.random.get_rng_state().clone()
    target = ContinuousJointCore(dict(base, contextual_message=True))
    assert torch.equal(rng, torch.random.get_rng_state())
    old, new = source.state_dict(), target.state_dict()
    extra = set(new) - set(old)
    assert len(extra) == 40 and all('.pair_message.' in name for name in extra)
    assert not (set(old) - set(new))
    target.load_state_dict({**new, **old}, strict=True)
    assert ContinuousJointCore.from_config(target.get_config()).get_config() == target.get_config()


def test_live_and_frozen_coefficients_drive_actual_residual():
    from honf_forward_core.continuous_joint.preparation import contextual_value_update, FrozenContextPlan, FrozenSDPAContextPlan

    torch.manual_seed(11)
    route = SpatialRouteAttentionV2(128, 4, 2, contextual_message=True)
    route.capture_pair_diagnostics = True
    with torch.no_grad():
        route.pair_message.output.normal_(std=.1)
    q = torch.randn(1, 4, 2, 32, requires_grad=True)
    k = torch.randn(1, 4, 3, 32, requires_grad=True)
    v = torch.randn(1, 4, 3, 32, requires_grad=True)
    receiver = torch.randn(1, 2, 128, requires_grad=True)
    donor = torch.randn(1, 3, 128, requires_grad=True)
    geometry = torch.randn(1, 2, 3, 5, requires_grad=True)
    bias = torch.randn(1, 4, 2, 3, requires_grad=True)
    allowed = torch.tensor([[[True, True, False], [False, False, False]]])
    measure = torch.ones(1, 3)
    mask = torch.tensor([[True, False]])
    kwargs = dict(route_name='em', stage_identity='cycle0/em', destination_ids=None, donor_ids=None,
                  receiver_normalized=receiver, donor_normalized=donor, geometry_features=geometry)
    gate = torch.tensor([[[.4, .8, 0], [0, 0, 0]]], requires_grad=True)
    class Policy:
        execution_backend = 'segment'
        def __call__(self, *args, **kwargs):
            return gate
    policy = Policy()
    out, weights, incidence, mass = contextual_value_update(route, q, k, v, bias, measure, allowed, mask, policy, **kwargs)
    assert out[:, 1].count_nonzero() == 0
    trace = route.pair_diagnostics
    assert trace['stage'] == 'cycle0/em' and trace['positive_edge_heads'] == 8
    torch.testing.assert_close(trace['base'] + trace['correction'], out.detach())
    out.sum().backward(retain_graph=True)
    assert trace['backward_edge_heads'] == 8
    assert gate.grad.abs().sum() > 0 and q.grad.abs().sum() > 0
    plan = FrozenContextPlan(incidence.detach(), weights.permute(0, 2, 3, 1).detach(), mass.detach())
    class Frozen:
        def __call__(self, *args, **kwargs): return plan
    frozen = contextual_value_update(route, q, k, v, bias, measure, allowed, mask, Frozen(), **kwargs)[0]
    torch.testing.assert_close(out, frozen)
    class Open:
        all_open = True
        def __call__(self, scores, allowed, *args, **kwargs): return allowed.to(scores.dtype)
    opened = contextual_value_update(route, q, k, v, bias, measure, allowed, mask, Open(), **kwargs)[0]
    additive = bias.detach().masked_fill(~allowed[:, None], -torch.inf)
    additive = torch.where(mask[:, None, :, None], additive, torch.zeros_like(additive))
    frozen_sdpa = FrozenSDPAContextPlan(allowed, q.detach(), k.detach(), additive, mask, mask[..., None].expand(-1, -1, 4).float())
    class FrozenSDPA:
        def __call__(self, *args, **kwargs): return frozen_sdpa
    replay = contextual_value_update(route, q, k, v, bias, measure, allowed, mask, FrozenSDPA(), **kwargs)[0]
    torch.testing.assert_close(opened, replay)
