"""Corrective equations and executor behavior; real learning is measured separately."""
from dataclasses import replace

import pytest
import torch

from tests.test_interaction_reused_member_routing import _base_config, _scene, _query
from honf_forward_core.interaction_reused_member_routing import RelativeMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.relative import (
    relative_scores, read_probability, mass_prefix_mask,
)


def _model(tile=16):
    return RelativeMemberRoutingCore(_base_config(), query_tile_size=tile)


def test_relative_scores_remove_offsets_and_bound_scale_changes():
    scores = torch.tensor([[[0.1, 0.4, 1.2, -8.]]], dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[[True, True, True, False]]])
    reference = relative_scores(scores, mask)
    for offset in (-100., 12., 100.):
        torch.testing.assert_close(relative_scores(scores + offset, mask), reference)
    for scale in (0., .01, 1., 1000.):
        varied = relative_scores(scores * scale, mask)
        assert torch.isfinite(varied).all() and varied.abs().max() <= 4
        assert torch.isfinite(torch.autograd.grad(varied.sum(), scores, retain_graph=True)[0]).all()
    flat = torch.zeros(1, 1, 3, dtype=torch.double, requires_grad=True)
    normalized = relative_scores(flat, torch.ones_like(flat, dtype=torch.bool))
    gradient = torch.autograd.grad(normalized[..., 0].sum(), flat)[0]
    torch.testing.assert_close(gradient, torch.tensor([[[2/3, -1/3, -1/3]]], dtype=torch.double))


def test_read_strength_has_actual_recovery_derivative_including_zero():
    contrast = torch.tensor([-100., -20., -1., 0., 1., 20., 100.], dtype=torch.double, requires_grad=True)
    probability = read_probability(contrast)
    derivative = torch.autograd.grad(probability.sum(), contrast)[0]
    expected = .5 / torch.hypot(contrast, torch.ones_like(contrast)).pow(3)
    torch.testing.assert_close(derivative, expected)
    assert (derivative > 0).all() and ((probability > 0) & (probability < 1)).all()
    assert torch.autograd.gradcheck(read_probability, (contrast.detach().requires_grad_(),))


def test_sparse_mass_budget_and_original_identity_ties_allow_empty_reads():
    weights = torch.tensor([[[.2, .2, .1], [.001, .002, .003]]])
    allowed = torch.ones_like(weights, dtype=torch.bool)
    ids = torch.tensor([[30, 10, 20]])
    mask = mass_prefix_mask(weights, allowed, ids, .11)
    assert torch.equal(mask, torch.tensor([[[True, True, False], [False, False, False]]]))
    assert ((weights * ~mask).sum(-1) <= .11 + 1e-7).all()
    permutation = torch.tensor([2, 0, 1])
    permuted = mass_prefix_mask(weights[..., permutation], allowed[..., permutation], ids[:, permutation], .3)
    original = mass_prefix_mask(weights, allowed, ids, .3)
    assert torch.equal(permuted, original[..., permutation])


def test_core_relative_offsets_live_gradients_and_continuous_sparse_accounting():
    torch.manual_seed(18)
    core = _model()
    prepared = core.prepare(_scene())
    query = _query(4)
    output = core.decode(prepared, query, return_trace=True)
    trace = output.trace
    controls = {f"fixed_member_logits_{role}": trace[f"member_logits_{role}"] + 30.
                for role in ("M", "E")}
    shifted = core.decode(prepared, query, return_trace=True, intervention=controls)
    torch.testing.assert_close(output.joint_fields, shifted.joint_fields, atol=2e-6, rtol=2e-5)
    targets = (prepared.B_E, prepared.member_keys_E, prepared.member_values_E,
               trace["member_read_strength_E"], trace["member_logits_E"])
    gradients = torch.autograd.grad(output.joint_fields.square().mean(), targets, allow_unused=True)
    assert all(value is not None and torch.isfinite(value).all() and value.abs().max() > 0 for value in gradients)
    assert output.work_counts["query_E_continuous_value_rows"] == int(trace["candidate_mask_E"].sum())
    with core.read_execution("sparse"):
        sparse = core.decode(prepared, query, return_trace=True)
    assert (sparse.trace["member_omitted_mass_E"] <= core.sparse_mass_tolerance + 1e-6).all()
    assert sparse.work_counts["query_E_continuous_value_rows"] == 0
    assert sparse.work_counts["query_E_sorted_rows"] == trace["candidate_mask_E"].numel()
    assert torch.equal(trace["original_environment_ids"], sparse.trace["original_environment_ids"])
    assert RelativeMemberRoutingCore.from_config(core.get_config()).get_config() == core.get_config()


def test_empty_type_and_closed_sparse_read_are_legitimate_without_nan_gradients():
    torch.manual_seed(19)
    core = _model()
    scene = _scene()
    scene = replace(scene, environment_measure=torch.zeros_like(scene.environment_measure))
    prepared = core.prepare(scene)
    output = core.decode(prepared, _query(3), return_trace=True)
    assert not output.trace["candidate_mask_E"].any()
    assert torch.equal(output.trace["member_null_E"], torch.ones(1, 3))
    output.joint_fields.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in core.parameters() if p.grad is not None)
    core.zero_grad(set_to_none=True)
    core.eval()
    with torch.no_grad():
        for gate in core.typed_read_gates.values():
            gate[-1].weight.zero_()
            gate[-1].bias.copy_(torch.tensor([-20., 20.]))
    sparse = core.decode(core.prepare(_scene()), _query(3), return_trace=True)
    assert sparse.work_counts["query_M_selected_value_rows"] == 0
    assert sparse.work_counts["query_E_selected_value_rows"] == 0
    assert (sparse.trace["member_read_strength_E"] > 0).all()
    assert (sparse.trace["member_gate_derivative_E"] > 0).all()


def test_routing_feature_normalization_controls_block_scale_and_checkpoint_replay():
    core = _model(tile=2)
    features = torch.randn(2, 3, 16)
    torch.testing.assert_close(core._routing_features(features), core._routing_features(features * 100.),
                               atol=5e-5, rtol=5e-5)
    core.train()
    output = core.decode(core.prepare(_scene()), _query(5))
    output.joint_fields.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in core.parameters() if p.grad is not None)
    assert output.work_counts["query_M_continuous_value_rows"] == 15
    assert output.work_counts["query_address_softmax_rows"] == 15


def test_independent_typed_skip_and_absent_addresses_preserve_access_semantics():
    core = _model()
    core.eval()
    with torch.no_grad():
        for gate in core.typed_read_gates.values():
            gate[-1].weight.zero_()
            gate[-1].bias.zero_()
        core.typed_read_gates["M"][-1].bias.copy_(torch.tensor([-20., 20.]))
    output = core.decode(core.prepare(_scene()), _query(3), return_trace=True)
    assert output.work_counts["query_M_selected_value_rows"] == 0
    assert output.work_counts["query_E_selected_value_rows"] > 0
    assert (output.trace["proposal_weights_M"].sum(-1) > 0).all()
    empty = core.decode(core.prepare(_scene(module_count=0)), _query(3), return_trace=True)
    assert empty.work_counts["query_M_selected_value_rows"] == 0
    assert empty.work_counts["query_E_selected_value_rows"] == 0
    assert torch.isfinite(empty.joint_fields).all()
