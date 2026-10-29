"""Opt-in active-domain access, original-record mass and native replay."""
import copy
import io
from dataclasses import fields, replace

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing.anchored import (
    AnchoredMemberRoutingCore, compile_e_access_directory,
)
from tests.test_r13_11_relative_ee import make_model


def compile_row(em, ee, mm, me, mask_m=None, mask_e=None):
    em, ee = torch.tensor([em], dtype=torch.double), torch.tensor([ee], dtype=torch.double)
    mm, me = torch.tensor([mm], dtype=torch.double), torch.tensor([me], dtype=torch.double)
    mask_m = torch.ones_like(mm, dtype=torch.bool) if mask_m is None else torch.tensor([mask_m])
    mask_e = torch.ones_like(me, dtype=torch.bool) if mask_e is None else torch.tensor([mask_e])
    return compile_e_access_directory(em, ee, mm, me, mask_m, mask_e,
        torch.arange(mm.shape[1])[None], torch.arange(me.shape[1])[None],
        access_measure='active_domain', retained_mass=0.5)


def test_active_measure_split_conservation_empty_roles_and_structural_self():
    # The last E slot is padded, the second M has zero measure, and an E
    # donor is split into two identical gates with measures 1 + 2 = 3.
    base = compile_row([[2., 5.], [0., 0.], [4., 4.]],
        [[0., 2., 7.], [0., 0., 7.], [7., 7., 7.]], [1., 0.], [1., 3., 0.],
        mask_e=[True, True, False])
    split = compile_row([[2., 5.], [0., 0.], [0., 0.], [4., 4.]],
        [[0., 2., 2., 7.], [0., 0., 0., 7.], [0., 0., 0., 7.], [7., 7., 7., 7.]],
        [1., 0.], [1., 1., 2., 0.], mask_e=[True, True, True, False])
    row = base.full.row(0, 0); sr = split.full.row(0, 0)
    def mass(r, indexes):
        return r.access_weight[torch.isin(r.member_index, torch.tensor(indexes))].sum()
    torch.testing.assert_close(mass(row, [0]), mass(sr, [0]))
    torch.testing.assert_close(mass(row, [2]), mass(sr, [2]))
    torch.testing.assert_close(mass(row, [3]), mass(sr, [3, 4]))
    assert 1 not in row.member_index.tolist() and 4 not in row.member_index.tolist()
    assert base.full.row(0, 2).member_index.numel() == 0
    empty_m = base.full.row(0, 1)
    assert empty_m.member_index.tolist() == [3]
    assert empty_m.structural_self.tolist() == [1.]
    restricted = base.restricted.row(0, 0)
    self_entry = restricted.member_index == 2
    assert self_entry.any() and restricted.structural_self[self_entry].item() == 1.


def test_zero_gate_has_zero_boundary_derivative_outside_active_domain():
    em = torch.tensor([[[2., 0.]]], dtype=torch.double, requires_grad=True)
    ee = torch.zeros((1, 1, 1), dtype=torch.double)
    directory = compile_e_access_directory(
        em, ee, torch.ones((1, 2), dtype=torch.double), torch.ones((1, 1), dtype=torch.double),
        torch.ones((1, 2), dtype=torch.bool), torch.ones((1, 1), dtype=torch.bool),
        torch.tensor([[0, 1]]), torch.tensor([[0]]), access_measure='active_domain',
    )
    row = directory.full.row(0, 0)
    mass = row.access_weight[row.member_index == 0].sum()
    gradient, = torch.autograd.grad(mass, em)
    assert gradient[0, 0, 0] != 0
    assert gradient[0, 0, 1] == 0


@pytest.mark.parametrize('locator', ['e_anchor', 'local_e_cover'])
def test_active_serialized_native_replay_and_actual_relation_reuse(locator):
    model, scene, query = make_model(locator, 'relative_quadratic')
    model.access_measure = 'active_domain'
    prepared = model.prepare(scene, trace_request={'include': []})
    assert len(prepared.context_stage_traces) == 2
    for i, stage in enumerate(prepared.context_stage_traces):
        for route in ('em', 'ee'):
            snap = stage[f'{route}_native_route_snapshot']
            assert snap.stage_identity == f'cycle{i}/{route}'
            assert snap.applied_incidence is stage[f'{route}_applied_incidence']
    for route in ('em', 'ee'):
        assert getattr(prepared, f'anchored_applied_{route.upper()}') is prepared.context_stage_traces[-1][f'{route}_applied_incidence']
    for b in range(scene.batch_size):
        for anchor in range(scene.num_environment_tokens):
            row = prepared.e_access_directory.full.row(b, anchor)
            for role, block in ((True, prepared.anchored_applied_EM), (False, prepared.anchored_applied_EE)):
                take = row.is_module == role
                ids = row.member_index[take] - (0 if role else scene.num_modules)
                torch.testing.assert_close(row.applied_exchange[take], block[b, anchor, ids])
    assert model.get_config()['access_measure'] == 'active_domain'
    buf = io.BytesIO(); torch.save({'config': model.get_config(), 'state': model.state_dict()}, buf); buf.seek(0)
    saved = torch.load(buf, weights_only=True)
    restored = AnchoredMemberRoutingCore.from_config(saved['config']).double().eval()
    restored.load_state_dict(saved['state'])
    for x, y in zip(model.decode_paired(prepared, query)[:2], restored.decode_paired(restored.prepare(scene), query)[:2]):
        torch.testing.assert_close(x.joint_fields, y.joint_fields, rtol=0, atol=0)
    old_config = dict(saved['config']); old_config.pop('access_measure')
    old = AnchoredMemberRoutingCore.from_config(old_config)
    assert old.access_measure == 'inventory_unit_total' and 'access_measure' not in old.get_config()
    old_config['access_measure'] = 'mean_one'
    assert AnchoredMemberRoutingCore.from_config(old_config).get_config()['access_measure'] == 'mean_one'


@pytest.mark.parametrize('locator', ['e_anchor', 'local_e_cover'])
def test_full_read_float64_output_and_all_input_parameter_gradients(locator):
    baseline, scene, query = make_model(locator, 'relative_quadratic')
    active = copy.deepcopy(baseline); active.access_measure = 'active_domain'
    def run(model):
        inputs = {f.name: v.detach().clone().requires_grad_(True) for f in fields(scene)
                  if torch.is_tensor(v := getattr(scene, f.name)) and v.is_floating_point()}
        live_scene = replace(scene, **inputs)
        coords = query.coordinates.detach().clone().requires_grad_(True)
        weights = (torch.stack((coords[..., 0].sigmoid(), 1-coords[..., 0].sigmoid()), -1)
                   if locator == 'local_e_cover' else query.anchor_weights)
        live_query = replace(query, coordinates=coords, anchor_weights=weights)
        output = model.decode(model.prepare(live_scene), live_query, intervention={'read_mode': 'full'}).joint_fields
        gradients = torch.autograd.grad(output.square().sum(), list(inputs.values()) + [coords] + list(model.parameters()), allow_unused=True)
        return output, gradients
    out0, grad0 = run(baseline); out1, grad1 = run(active)
    torch.testing.assert_close(out0, out1, rtol=1e-10, atol=1e-11)
    for a, b in zip(grad0, grad1):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize('locator', ['e_anchor', 'local_e_cover'])
def test_active_paired_checkpoint_backward_replay(locator, monkeypatch):
    import tests.test_r13_11_relative_ee as relative
    original = relative.make_model
    def active_model(*args):
        model, scene, query = original(*args)
        model.access_measure = 'active_domain'
        return model, scene, query
    monkeypatch.setattr(relative, 'make_model', active_model)
    relative.test_training_checkpoint_replays_paired_reads_and_preserves_gradients(
        'relative_quadratic', locator, monkeypatch)
