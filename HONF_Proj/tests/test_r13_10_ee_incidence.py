"""EE specialist-head aggregation and live contextual reuse."""
import copy
import io
from dataclasses import fields, replace

import pytest
import torch
from torch import nn

from honf_forward_core.continuous_joint.preparation import contextual_value_update
from honf_forward_core.continuous_joint.primitives import positive_quadratic_gate
from honf_forward_core.interaction_reused_member_routing.core import AppliedInteractionIncidence
from honf_forward_core.interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore
from tests.test_e_anchored_packed_execution import fixture


def test_post_gate_reference_masks_empty_permutation_and_containment():
    policy = AppliedInteractionIncidence(2).double()
    policy.ee_incidence_aggregation = "post_gate"
    scores = torch.tensor([[[[0., -2.], [float('nan'), float('nan')]],
                            [[-4., 0.], [float('nan'), float('nan')]]]], dtype=torch.double)
    allowed = torch.tensor([[[True, True], [False, False]]])
    inputs = torch.tensor([[[[1.5, -0.5], [1.5, 1.5]],
                             [[-2.5, 1.5], [1.5, 1.5]]]], dtype=torch.double)
    expected = (positive_quadratic_gate(inputs).mean(1) * allowed)
    actual = policy(scores, allowed, "ee")
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0, 0].item() == pytest.approx(0.34615384615384615)
    policy.ee_incidence_aggregation = "pre_gate"
    old = policy(scores, allowed, "ee")
    assert old[0, 0, 0] == 0 and actual[0, 0, 0] > 0
    assert torch.all((old > 0) <= (actual > 0))
    policy.ee_incidence_aggregation = "post_gate"
    torch.testing.assert_close(policy(scores.flip(1), allowed, "ee"), actual)
    with torch.no_grad():
        policy.routes['ee'].head_logits.copy_(torch.tensor([1., -1.], dtype=torch.double))
    weighted = policy(scores, allowed, 'ee')
    with torch.no_grad():
        policy.routes['ee'].head_logits.copy_(policy.routes['ee'].head_logits.flip(0))
    torch.testing.assert_close(policy(scores.flip(1), allowed, 'ee'), weighted)
    empty = policy(torch.empty(1, 2, 3, 0, dtype=torch.double), torch.empty(1, 3, 0, dtype=torch.bool), "ee")
    assert empty.shape == (1, 3, 0)
    one = AppliedInteractionIncidence(1).double()
    raw = torch.tensor([[[[0., -1.]]]], dtype=torch.double)
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    before = one(raw, mask, "ee")
    one.ee_incidence_aggregation = "post_gate"
    torch.testing.assert_close(one(raw, mask, "ee"), before)
    # Other routes keep the original law under U.
    torch.testing.assert_close(policy(scores, allowed, "em"),
                               positive_quadratic_gate(policy.routes["em"](scores, allowed)) * allowed)


def test_post_gate_is_the_live_contextual_message_and_gradient():
    policy = AppliedInteractionIncidence(2).double()
    policy.ee_incidence_aggregation = "post_gate"
    attention = nn.Module()
    attention.heads, attention.head_width, attention.width = 2, 1, 2
    attention.output = nn.Linear(2, 2, bias=True).double()
    q = torch.tensor([[[[1.]], [[1.]]]], dtype=torch.double, requires_grad=True)
    k = torch.tensor([[[[0.], [-2.]], [[-4.], [0.]]]], dtype=torch.double, requires_grad=True)
    v = torch.tensor([[[[2.], [3.]], [[4.], [5.]]]], dtype=torch.double, requires_grad=True)
    geometry = torch.zeros(1, 2, 1, 2, dtype=torch.double)
    measure = torch.tensor([[1., 2.]], dtype=torch.double)
    allowed = torch.ones(1, 1, 2, dtype=torch.bool)
    update, coeff, captured, mass = contextual_value_update(
        attention, q, k, v, geometry, measure, allowed, torch.ones(1, 1, dtype=torch.bool), policy,
        route_name="ee", stage_identity="cycle0/ee", destination_ids=None, donor_ids=None)
    scores = q @ k.transpose(-2, -1)
    expected_gate = policy(scores, allowed, "ee")
    base = (scores + measure.log()[:, None, None]).softmax(-1)
    z = (base * expected_gate[:, None]).sum(-1, keepdim=True)
    expected_coeff = base * expected_gate[:, None] / (z + .001 * (1-z).square())
    reduced = (expected_coeff @ v).transpose(1, 2).reshape(1, 1, 2)
    expected_mass = (z / (z + .001 * (1-z).square())).squeeze(-1).transpose(1, 2)
    expected_update = torch.nn.functional.linear(reduced, attention.output.weight, None) + attention.output.bias * expected_mass.mean(-1, keepdim=True)
    torch.testing.assert_close(captured, expected_gate)
    torch.testing.assert_close(coeff, expected_coeff)
    torch.testing.assert_close(mass, expected_mass)
    torch.testing.assert_close(update, expected_update)
    variables = (q, k, v, *policy.routes['ee'].parameters(), *attention.parameters())
    actual_g = torch.autograd.grad(update.square().sum(), variables, retain_graph=True)
    expected_g = torch.autograd.grad(expected_update.square().sum(), variables)
    for actual, expected in zip(actual_g, expected_g):
        torch.testing.assert_close(actual, expected)
    assert actual_g[3].abs().sum() > 0


@pytest.mark.parametrize('locator', ['e_anchor', 'local_e_cover'])
def test_post_gate_reuse_paired_read_checkpoint_and_gradients(locator):
    model, scene, query = fixture(locator)
    model.applied_incidence.ee_incidence_aggregation = 'post_gate'
    model.query_tile_size = 2
    other = copy.deepcopy(model)
    scene = replace(scene, **{f.name: v.detach().clone().requires_grad_(True)
        for f in fields(scene) if torch.is_tensor(v := getattr(scene, f.name)) and v.is_floating_point()})
    other_scene = replace(scene, **{f.name: v.detach().clone().requires_grad_(True)
        for f in fields(scene) if torch.is_tensor(v := getattr(scene, f.name)) and v.is_floating_point()})
    coords2 = query.coordinates.detach().clone().requires_grad_(True)
    if locator == 'local_e_cover':
        weight2 = coords2[..., 0].sigmoid()
        query2 = replace(query, coordinates=coords2, anchor_weights=torch.stack((weight2, 1-weight2), -1))
    else:
        query2 = replace(query, coordinates=coords2)
    prepared = model.prepare(scene, trace_request={'include': []})
    for cycle, stage in enumerate(prepared.context_stage_traces):
        snapshot = stage['ee_native_route_snapshot']
        assert snapshot.stage_identity == f'cycle{cycle}/ee'
        gate = stage['ee_applied_incidence']
        assert snapshot.applied_incidence is gate
        torch.testing.assert_close(gate, model.applied_incidence(snapshot.raw_scores, snapshot.allowed, 'ee'))
        measure = snapshot.donor_measure
        safe_measure = torch.where(measure > 0, measure, torch.ones_like(measure))
        logits = snapshot.raw_scores + safe_measure.log()[:, None, None, :]
        from honf_forward_core.content_hypergraph.staged_attention_v2 import safe_masked_softmax
        weights = safe_masked_softmax(logits, snapshot.allowed[:, None], -1)
        z = (weights * gate[:, None]).sum(-1)
        torch.testing.assert_close(snapshot.applied_message_mass, (z / (z + .001 * (1-z).square())).transpose(1, 2))
    final_gate = prepared.context_stage_traces[-1]['ee_applied_incidence']
    assert prepared.anchored_applied_EE is final_gate
    assert prepared.anchored_route_provenance['ee']['stage_identity'] == 'cycle1/ee'
    for anchor in range(scene.num_environment_tokens):
        row = prepared.e_access_directory.full.row(0, anchor)
        is_e = ~row.is_module
        indices = row.member_index[is_e] - scene.num_modules
        torch.testing.assert_close(row.applied_exchange[is_e], final_gate[0, anchor, indices])
    full, restricted, _ = model.decode_paired(prepared, query)
    reference = other.prepare(other_scene)
    separate = [other.decode(reference, query2, intervention={'read_mode': mode}) for mode in ('full', 'restricted')]
    for a, b in zip((full, restricted), separate):
        torch.testing.assert_close(a.joint_fields, b.joint_fields, rtol=1e-10, atol=1e-11)
    scene_vars = [getattr(scene, f.name) for f in fields(scene) if torch.is_tensor(getattr(scene, f.name)) and getattr(scene, f.name).is_floating_point()]
    other_vars = [getattr(other_scene, f.name) for f in fields(other_scene) if torch.is_tensor(getattr(other_scene, f.name)) and getattr(other_scene, f.name).is_floating_point()]
    targets = scene_vars + [query.coordinates] + list(model.parameters())
    reference_targets = other_vars + [query2.coordinates] + list(other.parameters())
    actual = torch.autograd.grad(sum(o.joint_fields.square().sum() for o in (full, restricted)), targets, allow_unused=True)
    expected = torch.autograd.grad(sum(o.joint_fields.square().sum() for o in separate), reference_targets, allow_unused=True)
    for a, b in zip(actual, expected):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-9)


def test_post_gate_serialization_restores_native_reads_and_historical_default():
    original, scene, query = fixture('e_anchor')
    config = original.get_config()
    assert AnchoredMemberRoutingCore.from_config(config).applied_incidence.ee_incidence_aggregation == 'pre_gate'
    config['ee_incidence_aggregation'] = 'post_gate'
    model = AnchoredMemberRoutingCore.from_config(config).double().eval()
    model.load_state_dict(original.state_dict())
    buffer = io.BytesIO()
    torch.save({'model_config': model.get_config(), 'model_state': model.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = AnchoredMemberRoutingCore.from_config(saved['model_config']).double().eval()
    restored.load_state_dict(saved['model_state'])
    assert restored.get_config()['ee_incidence_aggregation'] == 'post_gate'
    assert restored.score_calibration == 'inherited'
    assert restored.applied_incidence.ee_incidence_aggregation == 'post_gate'
    a = model.decode_paired(model.prepare(scene), query)
    b = restored.decode_paired(restored.prepare(scene), query)
    for actual, expected in zip(a[:2], b[:2]):
        torch.testing.assert_close(actual.joint_fields, expected.joint_fields, rtol=0, atol=0)
