"""Experimental pair content preserves the inherited model and live exchange."""
import pytest
import torch

from honf_forward_core.interaction_reused_member_routing.pair_content import PairContent, PairContentDetailAnchoredCore
from tests.test_r14_query_relative_values import _model_scene_query


@pytest.mark.parametrize("backend", ["segment", "dense_weighted"])
def test_zero_residual_full_model_and_gradients(backend):
    old, scene, query = _model_scene_query()
    new = PairContentDetailAnchoredCore.from_legacy_config(old.get_config()).double().eval()
    new.load_legacy_state_dict(old.state_dict())
    old.context_execution_backend = new.context_execution_backend = backend
    old_p, new_p = old.prepare(scene), new.prepare(scene)
    for a, b in zip(old_p.context_stage_traces, new_p.context_stage_traces):
        torch.testing.assert_close(a.applied_incidence, b.applied_incidence, rtol=0, atol=0)
    for mode in ("full", "restricted"):
        a, b = getattr(old_p.e_access_directory, mode), getattr(new_p.e_access_directory, mode)
        for name in ("member_index", "access_weight", "applied_exchange", "offsets"):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
    a = old.decode(old_p, query).joint_fields
    b = new.decode(new_p, query).joint_fields
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward(retain_graph=True)
    b.square().sum().backward(retain_graph=True)
    for name, p in old.named_parameters():
        q = dict(new.named_parameters())[name]
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in new.named_parameters() if name.endswith("pair_content.U"))
    for cycle in new.context_blocks:
        for route in (cycle.mm, cycle.em, cycle.ee, cycle.me):
            with torch.no_grad():
                route.pair_content.U.fill_(0.1)
    new.zero_grad(set_to_none=True)
    new.decode(new.prepare(scene), query).joint_fields.square().sum().backward()
    for factor in ("P", "Q", "R"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for name, p in new.named_parameters() if f"pair_content.{factor}.weight" in name)
    restored = PairContentDetailAnchoredCore.from_config(new.get_config()).double().eval()
    restored.load_state_dict(new.state_dict(), strict=True)
    torch.testing.assert_close(new.decode(new.prepare(scene), query).joint_fields,
                               restored.decode(restored.prepare(scene), query).joint_fields, rtol=0, atol=0)


def test_empty_zero_measure_and_masked_donors():
    old, scene, query = _model_scene_query()
    model = PairContentDetailAnchoredCore.from_legacy_config(old.get_config()).double().eval()
    model.load_legacy_state_dict(old.state_dict())
    # Entire donor roles empty for direct route execution, including explicit
    # padding and zero physical measures; no residual may leak into a row.
    route = model.context_blocks[0].em
    dst, donor = torch.randn(1, 2, 16, dtype=torch.double), torch.randn(1, 3, 16, dtype=torch.double)
    xy1, xy2 = torch.randn(1, 2, 2, dtype=torch.double), torch.randn(1, 3, 2, dtype=torch.double)
    lengths1, lengths2 = torch.ones_like(xy1), torch.ones_like(xy2)
    mask1, mask2 = torch.tensor([[True, False]]), torch.tensor([[True, False, True]])
    for n in (0, 3):
        update, weights = route(dst, donor[:, :n], xy1, xy2[:, :n], lengths1, lengths2[:, :n],
            torch.zeros(1, n, dtype=torch.double), torch.ones(1, dtype=torch.double), mask2[:, :n], mask1,
            torch.ones(1, 2, dtype=torch.double), value_policy=model._context_value_policy(),
            route_name="em", stage_identity="cycle0/em", return_attention=True)
        assert update.eq(0).all() and weights.eq(0).all()


def test_wind_three_dimensional_legacy_fixture():
    from Case_WindFarm.tests.test_relation_anchored_wind import _adapter, _case

    adapter = _adapter((4, 2, 1))
    scene = adapter.make_scene(_case())
    points = torch.tensor([[[.6, .7, .9], [1.4, 1.2, 1.1]]], dtype=torch.double)
    query = adapter.make_query(scene, points)
    old = adapter.core
    new = PairContentDetailAnchoredCore.from_legacy_config(old.get_config()).double().eval()
    new.load_legacy_state_dict(old.state_dict())
    a = old.decode(old.prepare(scene), query).joint_fields
    b = new.decode(new.prepare(scene), query).joint_fields
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_pair_gather_filters_zero_measure_and_zero_coefficients():
    pair = PairContent(8, 2, 2, 17).double()
    destination = torch.randn(1, 1, 8, dtype=torch.double)
    donor = torch.randn(1, 3, 8, dtype=torch.double)
    geometry = torch.randn(1, 1, 3, 5, dtype=torch.double)
    coefficients = torch.tensor([[[[.4, .2], [.1, .1], [0., 0.]]]], dtype=torch.double)
    gates = torch.ones(1, 1, 3, dtype=torch.double)
    allowed = torch.ones(1, 1, 3, dtype=torch.bool)
    measure = torch.tensor([[1., 0., 1.]], dtype=torch.double)
    seen = []
    hook = pair.P.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape[0]))
    pair(destination, donor, geometry, coefficients, gates, allowed, measure)
    hook.remove()
    assert pair.last_positive_edges == 1 and seen == [1]
    pair(destination, donor, geometry, coefficients, gates, allowed, torch.zeros_like(measure))
    assert pair.last_positive_edges == 0


@pytest.mark.parametrize('backend', ['segment', 'dense_weighted'])
def test_reported_pair_work_counts_actual_gather_inputs(backend):
    old, scene, _ = _model_scene_query()
    model = PairContentDetailAnchoredCore.from_legacy_config(old.get_config()).double().eval()
    model.load_legacy_state_dict(old.state_dict())
    model.context_execution_backend = backend
    observed = {route: [] for route in ('mm', 'em', 'ee', 'me')}
    hooks = []
    for cycle in model.context_blocks:
        for route in observed:
            module = getattr(cycle, route).pair_content.P
            hooks.append(module.register_forward_pre_hook(
                lambda module, inputs, route=route: observed[route].append(inputs[0].shape[0])))
    prepared = model.prepare(scene)
    for hook in hooks:
        hook.remove()
    for route, calls in observed.items():
        edges = sum(calls)
        assert prepared.work_counts[f'context_pair_content_{route}_positive_edges'] == edges
        assert prepared.work_counts[f'context_pair_content_{route}_head_pairs'] == edges * model.num_heads
        assert prepared.work_counts[f'context_pair_content_{route}_rank_scalars'] == edges * model.num_heads * 8
    assert prepared.work_counts['context_pair_content_positive_edges'] == sum(map(sum, observed.values()))
