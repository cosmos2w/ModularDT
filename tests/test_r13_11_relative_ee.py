"""Relative EE gates: physical statistics, applied exchange and replay."""
import copy
import io
from dataclasses import fields, replace

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing.core import AppliedInteractionIncidence
from honf_forward_core.interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore
from honf_forward_core.continuous_joint.primitives import positive_quadratic_gate
from tests.test_e_anchored_packed_execution import fixture


@pytest.mark.parametrize('mode', ['relative_quadratic', 'relative_sigmoid'])
def test_relative_gate_weighted_formula_masks_and_gradients(mode):
    policy = AppliedInteractionIncidence(2).double()
    policy.set_relative_ee_gate(mode, 1.25)
    scores = torch.tensor([[[[2., 0., -4., float('nan')], [float('nan')]*4],
                            [[0., 1., -1., float('nan')], [float('nan')]*4]]], dtype=torch.double, requires_grad=True)
    allowed = torch.tensor([[[True, True, True, False], [False]*4]])
    measure = torch.tensor([[1., 3., 0., 100.]], dtype=torch.double)
    gate = policy(scores, allowed, 'ee', donor_measure=measure)
    values = scores[:, :, :1, :2]
    weights = torch.tensor([.25, .75], dtype=torch.double)
    mean = (values * weights).sum(-1, keepdim=True)
    variance = ((values - mean).square() * weights).sum(-1, keepdim=True)
    delta = (values.amax(-1, keepdim=True) - values) / (1 + variance).sqrt()
    x = 1 - delta / policy.routes['ee'].tau
    per_head = 2 * positive_quadratic_gate(x) if mode == 'relative_quadratic' else x.sigmoid() / torch.sigmoid(x.new_tensor(1.))
    torch.testing.assert_close(gate[:, :1, :2], per_head.mean(1))
    assert gate[0, 0, 2:].count_nonzero() == 0 and gate[0, 1].count_nonzero() == 0
    assert gate.min() >= 0 and gate.max() <= 1
    gate.sum().backward()
    assert torch.isfinite(scores.grad).all() and policy.routes['ee'].raw_tau.grad.abs() > 0
    assert set(dict(policy.routes['ee'].named_parameters())) == {'head_logits', 'raw_tau'}
    empty = policy(torch.empty(1, 2, 3, 0, dtype=torch.double), torch.empty(1, 3, 0, dtype=torch.bool), 'ee', donor_measure=torch.empty(1, 0, dtype=torch.double))
    assert empty.shape == (1, 3, 0)


def make_model(locator, mode):
    original, scene, query = fixture(locator)
    config = original.get_config()
    config.update(ee_incidence_aggregation='post_gate', ee_gate_mode=mode, ee_initial_tau=1.25)
    model = AnchoredMemberRoutingCore.from_config(config).double().eval()
    # Common initialized parameters retain identical names, values and RNG behavior.
    common = dict(original.named_parameters())
    for name, value in model.named_parameters():
        if name in common:
            torch.testing.assert_close(value, common[name], rtol=0, atol=0)
    return model, scene, query


def test_relative_tied_peak_and_continuous_tail():
    scores = torch.tensor([[[[0., 0., -10.]]]], dtype=torch.double)
    allowed = torch.ones(1, 1, 3, dtype=torch.bool)
    measure = torch.tensor([[1., 1., 1.]], dtype=torch.double)
    results = []
    for mode in ('relative_quadratic', 'relative_sigmoid'):
        policy = AppliedInteractionIncidence(1).double()
        policy.set_relative_ee_gate(mode, .2)
        results.append(policy(scores, allowed, 'ee', donor_measure=measure))
    for gate in results:
        torch.testing.assert_close(gate[0, 0, :2], torch.ones(2, dtype=torch.double))
    assert results[0][0, 0, 2] == 0 and results[1][0, 0, 2] > 0


def test_fp32_convex_peak_roundoff_is_legal_in_actual_contextual_reduction():
    from types import SimpleNamespace
    from honf_forward_core.continuous_joint.preparation import contextual_value_update

    policy = AppliedInteractionIncidence(4)
    policy.set_relative_ee_gate('relative_quadratic', 1.49)
    with torch.no_grad():
        policy.routes['ee'].head_logits.copy_(torch.tensor([0.021946797, -0.038930506, 0.152425826, 0.114050463]))
    attention = SimpleNamespace(width=4, heads=4, head_width=1, output=torch.nn.Linear(4, 4))
    query = torch.zeros(1, 4, 1, 1)
    allowed = torch.ones(1, 1, 1, dtype=torch.bool)
    kwargs = dict(route_name='ee', stage_identity='cycle0/ee',
                  destination_ids=torch.zeros(1, 1, dtype=torch.long),
                  donor_ids=torch.zeros(1, 1, dtype=torch.long))
    update, _, gates, _ = contextual_value_update(
        attention, query, query, torch.ones_like(query), torch.zeros_like(query),
        torch.ones(1, 1), allowed, allowed[..., 0], policy, **kwargs)
    assert float(gates.detach().max()) > 1
    assert torch.isfinite(update).all()
    with pytest.raises(ValueError, match='must lie'):
        contextual_value_update(attention, query, query, torch.ones_like(query), torch.zeros_like(query),
                                torch.ones(1, 1), allowed, allowed[..., 0],
                                lambda *a, **kw: torch.full((1, 1, 1), 1.01), **kwargs)


@pytest.mark.parametrize('mode', ['relative_quadratic', 'relative_sigmoid'])
def test_live_message_removal_accepts_physical_measure_metadata(mode):
    model, scene, _ = make_model('e_anchor', mode)
    original = model.prepare(scene, trace_request={'include': []})
    stage = original.context_stage_traces[0]
    snapshot = stage['ee_native_route_snapshot']
    b, receiver, donor = (int(v) for v in (stage['ee_applied_incidence'] > 0).nonzero()[0])
    removal = {'stage': snapshot.stage_identity, 'route': 'ee',
               'destination_id': int(snapshot.destination_ids[b, receiver]),
               'donor_id': int(snapshot.donor_ids[b, donor])}
    altered = model.prepare(scene, context_message_removal=removal, trace_request={'include': []})
    assert altered.context_stage_traces[0]['ee_applied_incidence'][b, receiver, donor] == 0
    assert model._context_message_removal is None


@pytest.mark.parametrize('mode', ['relative_quadratic', 'relative_sigmoid'])
@pytest.mark.parametrize('locator', ['e_anchor', 'local_e_cover'])
def test_both_applied_cycles_directory_and_native_replay(mode, locator):
    model, scene, query = make_model(locator, mode)
    prepared = model.prepare(scene, trace_request={'include': []})
    for cycle, stage in enumerate(prepared.context_stage_traces):
        snapshot = stage['ee_native_route_snapshot']
        gate = stage['ee_applied_incidence']
        assert snapshot.stage_identity == f'cycle{cycle}/ee' and snapshot.applied_incidence is gate
        expected = model.applied_incidence(snapshot.raw_scores, snapshot.allowed, 'ee', donor_measure=snapshot.donor_measure)
        torch.testing.assert_close(gate, expected)
    final = prepared.context_stage_traces[-1]['ee_applied_incidence']
    assert prepared.anchored_applied_EE is final
    for anchor in range(scene.num_environment_tokens):
        row = prepared.e_access_directory.full.row(0, anchor)
        select = ~row.is_module
        indices = row.member_index[select] - scene.num_modules
        torch.testing.assert_close(row.applied_exchange[select], final[0, anchor, indices])
        torch.testing.assert_close(row.structural_self[select], (indices == anchor).to(row.structural_self))
    output = model.decode_paired(prepared, query)
    buffer = io.BytesIO()
    torch.save({'config': model.get_config(), 'state': model.state_dict()}, buffer)
    buffer.seek(0)
    data = torch.load(buffer, weights_only=True)
    restored = AnchoredMemberRoutingCore.from_config(data['config']).double().eval()
    restored.load_state_dict(data['state'])
    again = restored.decode_paired(restored.prepare(scene), query)
    for a, b in zip(output[:2], again[:2]):
        torch.testing.assert_close(a.joint_fields, b.joint_fields, rtol=0, atol=0)


def test_continuous_native_dense_segment_output_input_and_parameter_gradient_parity():
    model, scene, query = make_model('e_anchor', 'relative_sigmoid')
    other = copy.deepcopy(model)
    model.context_execution_backend = 'segment'
    other.context_execution_backend = 'dense_weighted'
    scene1 = replace(scene, **{f.name: v.detach().clone().requires_grad_(True)
        for f in fields(scene) if torch.is_tensor(v := getattr(scene, f.name)) and v.is_floating_point()})
    scene2 = replace(scene1, **{f.name: v.detach().clone().requires_grad_(True)
        for f in fields(scene1) if torch.is_tensor(v := getattr(scene1, f.name)) and v.is_floating_point()})
    out1 = model.decode_paired(model.prepare(scene1), query)
    out2 = other.decode_paired(other.prepare(scene2), query)
    for a, b in zip(out1[:2], out2[:2]):
        torch.testing.assert_close(a.joint_fields, b.joint_fields, rtol=2e-10, atol=2e-11)
    variables1 = [getattr(scene1, f.name) for f in fields(scene1) if torch.is_tensor(getattr(scene1, f.name)) and getattr(scene1, f.name).is_floating_point()] + list(model.parameters())
    variables2 = [getattr(scene2, f.name) for f in fields(scene2) if torch.is_tensor(getattr(scene2, f.name)) and getattr(scene2, f.name).is_floating_point()] + list(other.parameters())
    grads1 = torch.autograd.grad(sum(x.joint_fields.square().sum() for x in out1[:2]), variables1, allow_unused=True)
    grads2 = torch.autograd.grad(sum(x.joint_fields.square().sum() for x in out2[:2]), variables2, allow_unused=True)
    for a, b in zip(grads1, grads2):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize('mode', ['relative_quadratic', 'relative_sigmoid'])
@pytest.mark.parametrize('locator', ['e_anchor', 'local_e_cover'])
def test_training_checkpoint_replays_paired_reads_and_preserves_gradients(mode, locator, monkeypatch):
    import honf_forward_core.interaction_reused_member_routing.anchored as anchored

    model, scene, query = make_model(locator, mode)
    model.train()
    model.query_tile_size = 2
    reference = copy.deepcopy(model)
    checkpoint_impl = anchored.checkpoint
    calls = []

    def recorded_checkpoint(function, *args, **kwargs):
        def record_execution(*values):
            calls.append(1)
            return function(*values)
        return checkpoint_impl(record_execution, *args, **kwargs)

    def run(current):
        inputs = {f.name: v.detach().clone().requires_grad_(True)
                  for f in fields(scene) if torch.is_tensor(v := getattr(scene, f.name))
                  and v.is_floating_point()}
        live_scene = replace(scene, **inputs)
        coordinates = query.coordinates.detach().clone().requires_grad_(True)
        weights = (torch.stack((coordinates[..., 0].sigmoid(), 1 - coordinates[..., 0].sigmoid()), -1)
                   if locator == 'local_e_cover' else query.anchor_weights)
        live_query = replace(query, coordinates=coordinates, anchor_weights=weights)
        outputs = current.decode_paired(current.prepare(live_scene), live_query)[:2]
        before_backward = len(calls)
        targets = list(inputs.values()) + [coordinates] + list(current.parameters())
        gradients = torch.autograd.grad(sum(o.joint_fields.square().sum() for o in outputs), targets, allow_unused=True)
        return outputs, gradients, before_backward

    monkeypatch.setattr(anchored, 'checkpoint', recorded_checkpoint)
    actual, actual_gradients, forward_calls = run(model)
    assert forward_calls > 0 and len(calls) > forward_calls, 'backward must recompute checkpointed tiles'
    monkeypatch.setattr(anchored, 'checkpoint', lambda function, *args, **kwargs: function(*args))
    expected, expected_gradients, _ = run(reference)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.joint_fields, b.joint_fields, rtol=0, atol=0)
    for a, b in zip(actual_gradients, expected_gradients):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-9)
