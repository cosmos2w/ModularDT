"""Tiny positive permissions must have a usable organizer VJP after self exclusion.

These CPU algebra fixtures use no native checkpoint, physical model, or optimizer.
The small mass is intentional: the dominant module is the excluded MM self source.
"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import _membership, _pool
from honf_forward_core.interface_fields.overlap_control_hypergraph import (
    _masked_softmax,
    _source_density,
)
from honf_forward_core.interface_fields.receiver_tree_access import receiver_tree_access
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.typed_hypergraph_state import TypedSourceAccess, source_moments
from honf_forward_core.training.hypergraph_shadow import hard_value_soft_hypergraph_forward


@pytest.mark.parametrize("membership", [_membership, _source_density], ids=["tree", "overlap"])
def test_tiny_eligible_mm_mass_has_finite_score_and_access_vjp(membership):
    scores = torch.tensor([[[0., -86., -87.], [0., -87., -86.]]], requires_grad=True)
    access_logits = torch.tensor([[[-.1, .2]]], requires_grad=True)
    controls = torch.tensor([[[100.], [-100.]]], requires_grad=True)
    measures = torch.ones(1, 3)
    valid = torch.ones(1, 3, dtype=torch.bool)
    eligible = torch.tensor([[[False, True, True]]])
    edge = _masked_softmax(access_logits, torch.ones_like(access_logits, dtype=torch.bool), 1.)
    member = membership(scores, measures, soft=True, temperature=1.)
    actual = source_moments(edge, member, controls, measures, valid, pair_valid=eligible)
    adjoint = torch.tensor([[[0., .3, -.7]]])
    loss = (actual.weight * actual.control[..., 0] * adjoint).sum()
    assert torch.isfinite(loss)
    assert actual.support.tolist() == [[[False, True, True]]]
    gradients = torch.autograd.grad(loss, (scores, access_logits, controls))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.dtype == torch.float32
        assert gradient.abs().sum() > 0

    # Independent collapsed source-union adjoint: weight * conditional control
    # cancels the pair density. This avoids the problematic intermediate VJP.
    s, a, c = [value.detach().double().requires_grad_() for value in (scores, access_logits, controls)]
    probability = torch.softmax(s, -1)
    edge_probability = torch.softmax(a, -1)
    density = (edge_probability[..., None] * (3. * probability)[:, None]).sum(-2)
    moment = (edge_probability[..., None] * (3. * probability)[:, None]
              * c[:, None, :, 0, None]).sum(-2)
    density = density[..., 1:]
    moment = moment[..., 1:]
    expected_loss = (2. * moment * adjoint.double()[..., 1:]).sum() / density.sum()
    expected_gradients = torch.autograd.grad(expected_loss, (s, a, c))
    torch.testing.assert_close(loss.double(), expected_loss, rtol=3e-6, atol=1e-6)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient.double(), expected_gradient, rtol=3e-5, atol=1e-5)


@pytest.mark.parametrize("membership", [_membership, _source_density], ids=["tree", "overlap"])
def test_soft_membership_preserves_fp32_restoration_floor_and_source_masks(membership):
    scores = torch.tensor([[[0., -1000., 1000.]]], requires_grad=True)
    measures = torch.tensor([[1., 1., 0.]])
    member = membership(scores, measures, soft=True, temperature=1.)
    assert member[0, 0, 2] == 0
    torch.testing.assert_close(member[0, 0, 1].double(), torch.tensor(2. * torch.finfo(torch.float32).tiny, dtype=torch.float64), rtol=1e-6, atol=0)
    torch.testing.assert_close(member[0, 0, :2].mean(), member.new_tensor(1.), rtol=1e-6, atol=0)
    torch.autograd.grad(member.sum(), scores)


def test_soft_access_floor_invalid_empty_rows_and_mm_self_are_preserved():
    logits = torch.tensor([[[0., -1000., 1000.], [1., 2., 3.]]], requires_grad=True)
    valid = torch.tensor([[[True, True, False], [False, False, False]]])
    access = _masked_softmax(logits, valid, 1.)
    torch.testing.assert_close(access[0, 0, 1].double(), torch.tensor(torch.finfo(torch.float32).tiny, dtype=torch.float64), rtol=1e-6, atol=0)
    assert torch.count_nonzero(access[0, 1]) == 0
    assert access[0, 0, 2] == 0
    pair_valid = torch.tensor([[[False, True, False], [False, False, False]]])
    result = source_moments(access, torch.eye(3)[None], torch.tensor([[[100.], [-100.], [25.]]]), torch.tensor([[1., 1., 0.]]), torch.tensor([[True, True, False]]), pair_valid=pair_valid)
    assert result.support.tolist() == [[[False, True, False], [False, False, False]]]
    torch.testing.assert_close(result.weight[0, 0], result.weight.new_tensor([0., 1., 0.]), rtol=0, atol=0)
    assert torch.count_nonzero(result.control[0, 1]) == 0
    assert torch.count_nonzero(result.density[..., [0, 2]]) == 0
    gradient, = torch.autograd.grad(result.weight.sum() + result.control.sum(), logits)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[0, 1]) == 0
    empty = _masked_softmax(torch.zeros(1, 2, 0), torch.zeros(1, 2, 0, dtype=torch.bool), 1.)
    assert empty.shape == (1, 2, 0)


@pytest.mark.parametrize("membership", [_membership, _source_density], ids=["tree", "overlap"])
def test_empty_source_population_remains_zero_with_finite_vjp(membership):
    scores = torch.empty(1, 2, 0, requires_grad=True)
    logits = torch.tensor([[[-.1, .2]]], requires_grad=True)
    controls = torch.tensor([[[100.], [-100.]]], requires_grad=True)
    member = membership(scores, torch.empty(1, 0), soft=True, temperature=1.)
    edge = _masked_softmax(logits, torch.ones_like(logits, dtype=torch.bool), 1.)
    access = source_moments(edge, member, controls, torch.empty(1, 0), torch.empty(1, 0, dtype=torch.bool))
    assert access.weight.shape == (1, 1, 0)
    assert access.control.shape == (1, 1, 0, 1)
    assert access.support.numel() == 0
    inputs = (scores, logits, controls)
    gradients = torch.autograd.grad(access.weight.sum() + access.control.sum(), inputs, allow_unused=True)
    for source, gradient in zip(inputs, gradients):
        if gradient is None:
            # The existing overlap empty-axis branch returns zeros_like, with
            # no source-score graph to differentiate. This is legitimate.
            assert source is scores and source.numel() == 0
            continue
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


def test_hard_fp32_control_values_and_first_vjp_keep_literal_contract():
    edge = torch.tensor([[[.25, .75]]], requires_grad=True)
    controls = torch.tensor([[[100.], [-100.]]], requires_grad=True)
    access = source_moments(edge, torch.tensor([[[2., 0.], [0., 2.]]]), controls, torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool))
    for value in (access.density, access.weight, access.control):
        assert value.dtype == torch.float32
    torch.testing.assert_close(access.density, torch.tensor([[[.5, 1.5]]]), rtol=0, atol=0)
    torch.testing.assert_close(access.weight, torch.tensor([[[.5, 1.5]]]), rtol=0, atol=0)
    torch.testing.assert_close(access.control, torch.tensor([[[[100.], [-100.]]]]), rtol=0, atol=0)
    loss = (access.weight * access.control[..., 0] * torch.tensor([[[.5, -.25]]])).sum()
    edge_gradient, control_gradient = torch.autograd.grad(loss, (edge, controls))
    torch.testing.assert_close(edge_gradient, torch.tensor([[[37.5, -12.5]]]), rtol=0, atol=0)
    torch.testing.assert_close(control_gradient, torch.tensor([[[.25], [-.375]]]), rtol=0, atol=0)


@pytest.mark.parametrize("explicit_hard", [False, True], ids=["default", "explicit_hard"])
def test_hard_pool_preserves_mixed_dtype_population_mean_and_vjp(explicit_hard):
    states = torch.tensor([[[1.234567890123, -2.345678901234],
                            [-3.456789012345, 4.567890123456], [99., -99.]]],
                          dtype=torch.float64, requires_grad=True)
    mass = torch.tensor([[.123456789, .7654321, 0.]], requires_grad=True)
    result = _pool(states, mass, **({"soft_precision": False} if explicit_hard else {}))
    s, m = [value.detach().clone().requires_grad_() for value in (states, mass)]
    # Two-point population oracle. The original denominator is reduced in
    # mass precision, while the weighted states retain caller FP64 precision.
    expected = (s[:, 0] * m[:, 0, None] + s[:, 1] * m[:, 1, None]) / m.sum(-1, keepdim=True)
    widened = (s[:, 0] * m[:, 0, None] + s[:, 1] * m[:, 1, None]) / m.double().sum(-1, keepdim=True)
    assert not torch.equal(expected, widened), "Fixture must expose denominator widening."
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    adjoint = result.new_tensor([[.37, -1.21]])
    actual_vjp = torch.autograd.grad((result * adjoint).sum(), (states, mass))
    expected_vjp = torch.autograd.grad((expected * adjoint).sum(), (s, m))
    for actual, reference in zip(actual_vjp, expected_vjp):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert actual_vjp[0].dtype == torch.float64 and actual_vjp[1].dtype == torch.float32
    assert torch.count_nonzero(actual_vjp[0][:, 2]) == 0


def test_hard_source_moments_preserve_fp64_quadrature_near_and_first_vjp():
    edge = torch.tensor([[[.25, .75]]], requires_grad=True)
    member = torch.tensor([[[2., .125, 9.], [.375, 2., 7.]]], requires_grad=True)
    controls = torch.tensor([[[100., -11.], [-100., 23.]]], requires_grad=True)
    measures = torch.tensor([[.31234567890123, .91234567890123, 5.]], dtype=torch.float64, requires_grad=True)
    near = torch.tensor([[[.1234567890123, .4567890123456, .9]]], dtype=torch.float64, requires_grad=True)
    inputs = (edge, member, controls, measures, near)
    access = source_moments(edge, member, controls, measures,
                            torch.tensor([[True, True, False]]), near=near)
    a, p, c, mu, n = [value.detach().clone().requires_grad_() for value in inputs]
    # Explicit two-group/two-source union oracle: permission moments remain
    # FP32; native FP64 quadrature and near blending promote only afterwards.
    densities, conditional = [], []
    for source in (0, 1):
        first = a[..., 0] * p[:, None, 0, source]
        second = a[..., 1] * p[:, None, 1, source]
        density = first + second
        densities.append(density)
        conditional.append((first[..., None] * c[:, None, 0] + second[..., None] * c[:, None, 1]) / density[..., None])
    density = torch.stack(densities, -1)
    control = torch.stack(conditional, -2) * (1 - n[..., :2, None])
    mean = (mu[:, None, :2] * density).sum(-1, keepdim=True) / mu[:, None, :2].sum(-1, keepdim=True)
    weight = n[..., :2] + (1 - n[..., :2]) * density / mean
    assert access.density.dtype == torch.float32
    assert access.weight.dtype == access.control.dtype == torch.float64
    torch.testing.assert_close(access.density[..., :2], density, rtol=0, atol=0)
    torch.testing.assert_close(access.weight[..., :2], weight, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(access.control[..., :2, :], control, rtol=1e-14, atol=1e-14)
    assert access.support.tolist() == [[[True, True, False]]]
    assert access.weight[..., 2].item() == 0 and torch.count_nonzero(access.control[..., 2, :]) == 0
    adjoint = weight.new_tensor([[[[.37, -1.21], [2.4, .19]]]])
    actual_loss = (access.weight[..., :2, None] * access.control[..., :2, :] * adjoint).sum()
    expected_loss = (weight[..., None] * control * adjoint).sum()
    actual_vjp = torch.autograd.grad(actual_loss, inputs)
    expected_vjp = torch.autograd.grad(expected_loss, (a, p, c, mu, n))
    for actual, reference in zip(actual_vjp, expected_vjp):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
    assert actual_vjp[3][0, 2] == 0 and actual_vjp[4][0, 0, 2] == 0


class _TinyControlGain(nn.Module):
    """Only the physical Linear boundary; never constructs a field/backend."""

    def __init__(self, permission_mode):
        super().__init__()
        self.permission_mode = permission_mode
        self.plan_intervention = "joint"
        self.control_gain = nn.ModuleDict({"MM": nn.Linear(2, 1)})
        with torch.no_grad():
            self.control_gain["MM"].weight.copy_(torch.tensor([[.3125, -.6875]]))
            self.control_gain["MM"].bias.fill_(.0625)


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_modulate_cpu_bf16_preserves_hard_promotion_and_soft_boundary_vjp(mode):
    reader = _TinyControlGain(mode)
    reference = deepcopy(reader)
    messages = torch.tensor([[[[.8125, -1.375], [-.4375, 1.625]]]], dtype=torch.bfloat16, requires_grad=True)
    permission_dtype = torch.float64 if mode == "soft" else torch.float32
    weight = torch.tensor([[[1.23456789, .87654321]]], dtype=permission_dtype, requires_grad=True)
    control = torch.tensor([[[[.23456789, -.81234567], [-.72345678, .3456789]]]], dtype=permission_dtype, requires_grad=True)
    access = TypedSourceAccess(torch.ones_like(weight), weight, control, weight > 0, torch.empty(0))
    x, w, c = [value.detach().clone().requires_grad_() for value in (messages, weight, control)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = TypedHypergraphField._modulate(reader, messages, access, "MM")
        # Independent physical-reader expression with explicit completed
        # permission casts only for the soft policy path.
        completed_control = c.float() if mode == "soft" else c
        completed_weight = w.to(x.dtype) if mode == "soft" else w
        linear = reference.control_gain["MM"]
        gain = torch.nn.functional.linear(completed_control, linear.weight, linear.bias)
        expected = x * (1 + torch.tanh(gain)) * completed_weight[..., None]
        if mode == "hard":
            prematurely_cast = x * (1 + torch.tanh(gain)) * w.to(x.dtype)[..., None]
            assert not torch.equal(expected, prematurely_cast.float()), "Fixture must expose weight downcast."
    assert actual.dtype == (torch.float32 if mode == "hard" else torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    adjoint = actual.new_tensor([[[[.37, -1.21], [2.4, .19]]]])
    actual_inputs = (messages, weight, control, *reader.parameters())
    expected_inputs = (x, w, c, *reference.parameters())
    actual_vjp = torch.autograd.grad((actual * adjoint).sum(), actual_inputs)
    expected_vjp = torch.autograd.grad((expected * adjoint).sum(), expected_inputs)
    for actual_gradient, reference_gradient in zip(actual_vjp, expected_vjp):
        assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().sum() > 0
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)
    assert actual_vjp[1].dtype == actual_vjp[2].dtype == permission_dtype


def _one_split_geometry(boundary, overlap):
    return {
        "axes": torch.tensor([[0, 0, 0, 0]]),
        "parents": torch.tensor([[0, 0, 0, 0]]),
        "left": torch.tensor([[False, True, False, False]]),
        "depths": torch.tensor([[0, 1, 1, -1]]),
        "leaves": torch.tensor([[False, True, True, True]]),
        "valid": torch.tensor([[True, True, True, False]]),
        "boundary": boundary,
        "overlap": overlap,
        "spatial_dim": 2,
    }


@pytest.mark.parametrize("explicit_hard", [False, True], ids=["default", "explicit_hard"])
def test_hard_receiver_geometry_preserves_mixed_dtype_values_and_vjp(explicit_hard):
    query = torch.tensor([[[.2134567890123, .7], [-.10000000000003, -.1]]], dtype=torch.float64, requires_grad=True)
    gates = torch.tensor([[.43, .2, .7, .8]], requires_grad=True)
    boundary = torch.tensor([[.1234567, 0., 0., 0.]], requires_grad=True)
    overlap = torch.tensor([[.83, 1., 1., 1.]], requires_grad=True)
    data = _one_split_geometry(boundary, overlap)
    actual = receiver_tree_access(query, gates, data, 1, **({"soft_precision": False} if explicit_hard else {}))

    # Closed-form one-split tree oracle. Preserve the original expression's
    # FP32 boundary+half-width before subtracting FP64 caller coordinates.
    q, g, b, w = [value.detach().clone().requires_grad_() for value in (query, gates, boundary, overlap)]
    fraction = ((b[:, :1] + w[:, :1] / 2 - q[..., 0]) / w[:, :1]).clamp(0, 1)
    left = fraction.square() * (3 - 2 * fraction)
    root_gate = g[:, :1].square() * (3 - 2 * g[:, :1])
    expected = torch.stack(((1 - root_gate).expand_as(left), root_gate * left,
                            root_gate * (1 - left), torch.zeros_like(left)), -1)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    adjoint = actual.new_tensor([.37, -1.21, 2.4, 999.])
    gradients = torch.autograd.grad((actual * adjoint).sum(), (query, gates, boundary, overlap))
    references = torch.autograd.grad((expected * adjoint).sum(), (q, g, b, w))
    for gradient, reference in zip(gradients, references):
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
    assert gradients[0].dtype == torch.float64
    assert gradients[1].dtype == torch.float32
    assert torch.count_nonzero(actual[..., 3]) == 0
    assert gradients[1][0, 3] == 0


def test_soft_receiver_precision_is_explicit_and_retains_live_fp32_geometry():
    query = torch.tensor([[[.2134567, .7]]], requires_grad=True)
    gates = torch.tensor([[.43, .2, .7, .8]], dtype=torch.float64, requires_grad=True)
    boundary = torch.tensor([[.1234567, 0., 0., 0.]], requires_grad=True)
    overlap = torch.tensor([[.83, 1., 1., 1.]], requires_grad=True)
    result = receiver_tree_access(query, gates, _one_split_geometry(boundary, overlap), 1, soft_precision=True)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result.sum(-1), result.new_ones(1, 1), rtol=0, atol=1e-15)
    gradients = torch.autograd.grad((result * result.new_tensor([.37, -1.21, 2.4, 999.])).sum(), (query, gates, boundary, overlap))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert gradients[0].dtype == torch.float32
    assert gradients[1].dtype == torch.float64
    assert result[0, 0, 3] == 0 and gradients[1][0, 3] == 0


class _PermissionOnlyWrapper(nn.Module):
    """A scalar physical reader exercises the existing whole-wrapper boundary."""

    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.backend = nn.Module()
        self.core.backend.permission_mode = "hard"
        self.core.backend.organizer = nn.Module()
        organizer = self.core.backend.organizer
        organizer.scores = nn.Parameter(torch.tensor([[[0., -86., -87.], [0., -87., -86.]]]))
        organizer.access_logits = nn.Parameter(torch.tensor([[[-.1, .2]]]))
        organizer.controls = nn.Parameter(torch.tensor([[[100.], [-100.]]]))
        self.physical_gain = nn.Parameter(torch.tensor(2.))

    def forward(self, caller_input):
        backend = self.core.backend
        organizer = backend.organizer
        soft = backend.permission_mode == "soft"
        if soft:
            member = _membership(organizer.scores, torch.ones(1, 3), soft=True, temperature=1.)
            edge = _masked_softmax(organizer.access_logits, torch.ones_like(organizer.access_logits, dtype=torch.bool), 1.)
            access = source_moments(edge, member, organizer.controls, torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool), pair_valid=torch.tensor([[[False, True, True]]]))
            policy = (access.weight * access.control[..., 0] * torch.tensor([[[0., .3, -.7]]])).sum().to(caller_input.dtype)
        else:
            policy = caller_input.new_zeros(())
        return {"pred_field": self.physical_gain * (caller_input + policy)}


def test_tiny_permission_shadow_does_not_leak_into_physical_or_caller_gradients():
    model = _PermissionOnlyWrapper()
    reference = deepcopy(model)
    caller = torch.tensor(3., requires_grad=True)
    hard_caller = caller.detach().clone().requires_grad_()
    hard = reference(hard_caller)
    hard["pred_field"].square().backward()
    output = hard_value_soft_hypergraph_forward(model, caller)
    torch.testing.assert_close(output["pred_field"], hard["pred_field"], rtol=0, atol=0)
    assert output["pred_field"].dtype == torch.float32
    output["pred_field"].square().backward()
    torch.testing.assert_close(model.physical_gain.grad, reference.physical_gain.grad, rtol=0, atol=0)
    torch.testing.assert_close(caller.grad, hard_caller.grad, rtol=0, atol=0)
    for parameter in model.core.backend.organizer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.core.backend.permission_mode == "hard"
