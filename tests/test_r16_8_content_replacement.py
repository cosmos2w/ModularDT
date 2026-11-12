import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph.staged_attention_v2 import (
    ContextualContentReplacement, SpatialRouteAttentionV2,
)
from honf_forward_core.continuous_joint.preparation import contextual_value_update


@pytest.mark.parametrize('mode', ['c8', 's8'])
def test_replacement_matches_direct_equation_and_gradients(mode):
    torch.manual_seed(42)
    module = ContextualContentReplacement(6, 2, 3, mode).double()
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=.2)
    receiver = torch.randn(2, 3, 6, dtype=torch.double, requires_grad=True)
    donor = torch.randn(2, 4, 6, dtype=torch.double, requires_grad=True)
    geometry = torch.randn(2, 3, 4, 3, dtype=torch.double, requires_grad=True)
    value = torch.randn(2, 2, 4, 3, dtype=torch.double, requires_grad=True)
    eligible = torch.rand(2, 3, 4) > .4
    coefficients = (torch.rand(2, 3, 4, 2, dtype=torch.double) * eligible[..., None]).requires_grad_()
    actual = module(receiver, donor, geometry, value, coefficients, eligible, tile_size=3)
    shifts = []
    for sign in ('plus', 'minus'):
        shift = torch.einsum('bdc,hcv->bdhv', donor, getattr(module, 'q_'+sign))[:, None] + getattr(module, 'b_'+sign)
        if mode == 's8':
            shift = shift + torch.einsum('brc,hcv->brhv', receiver, getattr(module, 'p_'+sign))[:, :, None]
            shift = shift + torch.einsum('brdg,hgv->brdhv', geometry, getattr(module, 'r_'+sign))
        shifts.append(shift)
    v = value.transpose(1, 2)[:, None]
    expected = ((F.silu(v + shifts[0]) - F.silu(-v + shifts[1])) * coefficients[..., None]
                * eligible[..., None, None]).sum(2)
    torch.testing.assert_close(actual, expected)
    args = [donor, value, coefficients] + [p for p in module.parameters() if p.requires_grad]
    if mode == 's8':
        args += [receiver, geometry]
    a = torch.autograd.grad(actual.square().sum(), args, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), args)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-9)
    if mode == 'c8':
        assert all(not getattr(module, n+'_'+s).requires_grad and not getattr(module, n+'_'+s).count_nonzero()
                   for n in ('p', 'r') for s in ('plus', 'minus'))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('mode', ['c8', 's8'])
@pytest.mark.parametrize('all_open', [False, True])
def test_initial_identity_and_bias_mask(dtype, mode, all_open):
    torch.manual_seed(8)
    route = SpatialRouteAttentionV2(6, 2, 1).to(dtype)
    q, k, value = [torch.randn(1, 2, n, 3, dtype=dtype, requires_grad=True) for n in (3, 4, 4)]
    bias = torch.randn(1, 2, 3, 4, dtype=dtype, requires_grad=True)
    allowed = torch.tensor([[[True, True, False, False], [False]*4, [False]*4]])
    mask = torch.tensor([[True, True, False]])
    class Policy:
        def __call__(self, scores, allowed, *args, **kwargs):
            return allowed.to(dtype) * (1. if all_open else .4)
    policy = Policy()
    policy.all_open = all_open
    kw = dict(route_name='em', stage_identity='cycle0/em', destination_ids=None, donor_ids=None,
              receiver_normalized=torch.randn(1, 3, 6, dtype=dtype),
              donor_normalized=torch.randn(1, 4, 6, dtype=dtype),
              geometry_features=torch.randn(1, 3, 4, 3, dtype=dtype))
    args = (route, q, k, value, bias, torch.ones(1, 4, dtype=dtype), allowed, mask, policy)
    original, _, _, _ = contextual_value_update(*args, **kw)
    inherited = (q, k, value, bias, route.output.weight, route.output.bias)
    original_grad = torch.autograd.grad(original.square().sum(), inherited, retain_graph=True)
    route.enable_content_replacement(mode)
    actual, weights, _, mass = contextual_value_update(*args, **kw)
    replacement_grad = torch.autograd.grad(actual.square().sum(), inherited, retain_graph=True)
    expected = F.linear(torch.einsum('bhrd,bhdv->brhv', weights, value).reshape(1, 3, 6), route.output.weight, None)
    expected = (expected + route.output.bias * mass.mean(-1, keepdim=True)) * mask[..., None]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual, original, rtol=2e-5, atol=2e-6)
    for left, right in zip(original_grad, replacement_grad):
        torch.testing.assert_close(left, right, rtol=4e-5, atol=4e-6)
    assert actual[:, 1:].count_nonzero() == 0


def test_empty_edges_backward_and_identical_layout():
    a = ContextualContentReplacement(6, 2, 3, 'c8')
    b = ContextualContentReplacement(6, 2, 3, 's8')
    assert a.state_dict().keys() == b.state_dict().keys()
    receiver = torch.randn(1, 2, 6, requires_grad=True)
    donor = torch.randn(1, 3, 6, requires_grad=True)
    value = torch.randn(1, 2, 3, 3, requires_grad=True)
    result = b(receiver, donor, torch.zeros(1, 2, 3, 3), value,
               torch.zeros(1, 2, 3, 2), torch.zeros(1, 2, 3, dtype=torch.bool))
    assert result.count_nonzero() == 0
    result.sum().backward()
    assert value.grad is not None and all(p.grad is not None for p in b.parameters())


def test_initial_conditional_gradients_and_zero_value_message():
    torch.manual_seed(9)
    module = ContextualContentReplacement(6, 2, 3, 's8')
    assert all(p.count_nonzero() == 0 for p in module.parameters())
    receiver = torch.randn(1, 2, 6)
    donor = torch.randn(1, 3, 6)
    geometry = torch.randn(1, 2, 3, 3)
    value = torch.randn(1, 2, 3, 3)
    eligible = torch.ones(1, 2, 3, dtype=torch.bool)
    coefficients = torch.rand(1, 2, 3, 2)
    module(receiver, donor, geometry, value, coefficients, eligible).square().sum().backward()
    for name in ('p_plus', 'p_minus', 'r_plus', 'r_minus'):
        assert getattr(module, name).grad.abs().sum() > 0
    with torch.no_grad():
        module.b_plus.fill_(.5)
    out = module(receiver, donor, geometry, torch.zeros_like(value), coefficients, eligible)
    assert out.abs().sum() > 0


@pytest.mark.parametrize('mode', ['c8', 's8'])
def test_optional_work_counts_and_checkpoint_replays(mode):
    import copy

    torch.manual_seed(19)
    plain = ContextualContentReplacement(6, 2, 3, mode).double()
    tracked = copy.deepcopy(plain)
    tracked.capture_work = True
    receiver = torch.randn(1, 2, 6, dtype=torch.double, requires_grad=True)
    donor = torch.randn(1, 3, 6, dtype=torch.double, requires_grad=True)
    geometry = torch.randn(1, 2, 3, 3, dtype=torch.double, requires_grad=True)
    value = torch.randn(1, 2, 3, 3, dtype=torch.double, requires_grad=True)
    eligible = torch.tensor([[[True, False, True], [False, True, False]]])
    coefficients = torch.rand(1, 2, 3, 2, dtype=torch.double, requires_grad=True)
    args = receiver, donor, geometry, value, coefficients, eligible
    a = plain(*args, tile_size=2)
    b = tracked(*args, tile_size=2)
    torch.testing.assert_close(a, b)
    assert plain.work_trace is None
    work = tracked.work_trace
    assert work['donor_projection_rows'] == 6
    if mode == 'c8':
        assert work['donor_nonlinear_heads'] == 6
        assert work['coefficient_reduction_head_pairs'] == 12
        assert work['nonlinear_vector_elements'] == 36
        assert work['reduction_vector_elements'] == 36
        assert work['evaluated_edge_heads'] == 0
    else:
        assert work['receiver_projection_rows'] == 4
        assert work['geometry_projection_rows'] == 6
        assert work['evaluated_edge_heads'] == 6
        assert work['coefficient_reduction_head_pairs'] == 6
        assert work['nonlinear_vector_elements'] == 36
        assert work['reduction_vector_elements'] == 18
        assert work['forward_tile_calls'] == work['tile_calls'] == 2
        assert work['checkpoint_replay_calls'] == 0
    inputs = (receiver, donor, geometry, value, coefficients)
    ga = torch.autograd.grad(a.square().sum(), inputs, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(b.square().sum(), inputs, allow_unused=True)
    for x, y in zip(ga, gb):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y)
    assert work['checkpoint_replay_calls'] == (2 if mode == 's8' else 0)
    assert work['tile_calls'] == (4 if mode == 's8' else 0)


@pytest.mark.parametrize('mode', ['c8', 's8'])
def test_stages_and_actual_relation_reuse_in_paired_reads(mode):
    import copy
    from tests.test_r13_11_relative_ee import make_model

    baseline, scene, query = make_model('e_anchor', 'relative_quadratic')
    candidate = copy.deepcopy(baseline)
    routes = [(name, module) for name, module in candidate.named_modules()
              if isinstance(module, SpatialRouteAttentionV2)]
    order = []
    hooks = []
    for name, route in routes:
        route.enable_content_replacement(mode)
        route.content_replacement.capture_messages = True
        hooks.append(route.register_forward_hook(lambda mod, args, result, name=name: order.append(name.rsplit('.', 1)[-1])))
    original = baseline.prepare(scene, trace_request={'include': []})
    prepared = candidate.prepare(scene, trace_request={'include': []})
    assert order == ['mm', 'em', 'ee', 'me'] * 2
    for hook in hooks:
        hook.remove()
    for route in ('em', 'ee'):
        actual = getattr(prepared, f'anchored_applied_{route.upper()}')
        assert actual is prepared.context_stage_traces[-1][f'{route}_applied_incidence']
    for first, second in zip(original.context_stage_traces, prepared.context_stage_traces):
        for route in ('mm', 'em', 'ee', 'me'):
            torch.testing.assert_close(first[f'{route}_applied_incidence'], second[f'{route}_applied_incidence'])
    for left, right in zip(baseline.decode_paired(original, query)[:2], candidate.decode_paired(prepared, query)[:2]):
        torch.testing.assert_close(left.joint_fields, right.joint_fields, rtol=1e-9, atol=1e-10)
    for _, route in routes:
        trace = route.content_replacement.message_trace
        assert trace is not None and trace['messages'].shape[0] == trace['rows'].shape[0]
        assert trace['coefficients'].shape == trace['messages'].shape[:2]
