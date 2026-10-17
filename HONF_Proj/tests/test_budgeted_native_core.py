"""Round8 core integration: original boundary, live routing, and lifecycle."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore
from tests.test_content_hypergraph_v2_core import _core, _query, _scene


def _double(value):
    return replace(value, **{name: tensor.double() for name, tensor in vars(value).items()
                             if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()})


def test_original_boundary_and_optimizer_are_exact():
    parent = _core("A2-D").double().train()
    child = BudgetedNativeCore.from_a2d(parent)
    scene, query = _double(_scene(1)), _double(_query(1))
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001) for model in (parent, child)]
    for model, optimizer in zip((parent, child), optimizers):
        output = model.decode(model.prepare(scene), query, torch.ones(1, 3).double())
        loss = output.nonlinear_field.square().sum() + output.thermal_temperature.square().sum()
        loss.backward()
        optimizer.step()
    for name, tensor in parent.state_dict().items():
        assert torch.equal(tensor, child.state_dict()[name]), name
    assert all(parameter.grad is None for parameter in child.budgeted_router.parameters())


def test_dense_router_has_binary_forward_and_live_membership_dispatch_gradients():
    model = BudgetedNativeCore.from_a2d(_core("A2-D").double(), membership_bias=0., lambda0=.1).train()
    model.budgeted_begin_update(501, 0)
    scene, query = _double(_scene(1)), _double(_query(1))
    organization = model.prepare(scene)
    output = model.decode(organization, query, torch.ones(1, 3).double())
    result = model.budgeted_microbatch_result()
    work_penalty = model.budgeted_router.dual.penalty_residual(
        result["constraint_residual_numerator"] / result["work_penalty_denominator"])
    (output.nonlinear_field.square().sum() + output.thermal_temperature.square().sum()
     + work_penalty).backward()
    for name, parameter in model.budgeted_router.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.budgeted_router.membership.bias.grad.abs() > 0
    assert model.budgeted_router.dispatch.head_logits.grad.norm() > 0
    with pytest.raises(ValueError, match="consumed"):
        model.decode(organization, query)


def test_query_chunks_share_scene_memberships_and_counter_draws():
    model = BudgetedNativeCore.from_a2d(_core("A2-D").double(), membership_bias=0.).train()
    model.budgeted_begin_update(701, 0)
    scene, query = _double(_scene(1)), _double(_query(1))
    prepared = model.prepare(scene)
    full = model.decode(prepared, query).nonlinear_field
    blocks = []
    for a, b in ((0, 2), (2, 5)):
        part = replace(query, coordinates=query.coordinates[:, a:b], features=query.features[:, a:b],
                       query_ids=query.query_ids[:, a:b])
        blocks.append(model.decode(prepared, part).nonlinear_field)
    torch.testing.assert_close(torch.cat(blocks, 1), full, rtol=1e-12, atol=1e-12)


def test_global_macro_clock_and_dual_survive_exact_state_roundtrip():
    model = BudgetedNativeCore.from_a2d(_core("A2-D"), lambda0=.2).train()
    for update in range(10):
        model.budgeted_begin_update(701 + update // 5, update % 5)
        model._budgeted_observations = [(torch.tensor(.2), 1)]
        model.budgeted_after_update(701 + update // 5, update % 5)
    restored = BudgetedNativeCore.from_config(model.get_config())
    restored.load_state_dict(model.state_dict())
    assert restored._budgeted_macro_clock == 10
    assert restored._router_mode()[2] == model._router_mode()[2]
    for name, value in model.state_dict().items():
        other = restored.state_dict()[name]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other, name


def test_eval_heat_laws_and_graph_mutation_rejection():
    model = BudgetedNativeCore.from_a2d(_core("A2-D").double(), membership_bias=-1.).eval()
    model.set_native_training_epoch(2500)
    scene, query = _double(_scene(1)), _double(_query(1))
    with torch.no_grad():
        prepared = model.prepare(scene)
        h = torch.tensor([[1., -.5, 2.]], dtype=torch.double)
        output = model.decode(prepared, query, h, accumulation_dtype=torch.float64)
        zero = model.decode(prepared, query, h*0, accumulation_dtype=torch.float64)
        twice = model.decode(prepared, query, h*2, accumulation_dtype=torch.float64)
        assert torch.count_nonzero(zero.thermal_temperature) == 0
        assert torch.equal(output.nonlinear_field, twice.nonlinear_field)
        torch.testing.assert_close(twice.thermal_temperature, output.thermal_temperature*2,
                                   atol=1e-12, rtol=1e-12)
        prepared.native_hypergraph.membership_hard[0, 0, 1].logical_not_()
        with pytest.raises(ValueError, match="changed"):
            model.decode(prepared, query)


def test_only_mm_is_captured_and_no_environment_gate_exists():
    model = BudgetedNativeCore.from_a2d(_core("A2-D")).eval()
    model.set_native_training_epoch(2500)
    prepared = model.prepare(_scene(1))
    assert prepared.budgeted_route_snapshot.route == "mm"
    assert all("environment" not in name and ".me." not in name
               for name in model.budgeted_optimizer_parameter_names())
    assert prepared.work_counts["native_route_capture_score_rows"] == 4*3*3


def test_work_uses_unit_sampling_marginals_then_declared_sharp_relaxation():
    model = BudgetedNativeCore.from_a2d(_core("A2-D"))
    logits = torch.tensor([[[0., -1., .5], [1., 0., -.5], [.2, -.2, 0.]]])
    eye = torch.eye(3, dtype=torch.bool)[None]
    probability = torch.where(eye, torch.ones_like(logits), torch.sigmoid(logits))
    present = torch.ones(1, 3, dtype=torch.bool)
    qmask = torch.tensor([[True, False]])
    d = torch.tensor([[[.1, -.2, .3], [-torch.inf, -torch.inf, -torch.inf]]])
    dispatch = SimpleNamespace(dispatch_logits=d,
                               probabilities=torch.softmax(torch.nan_to_num(d), -1)*qmask[..., None])
    graph = SimpleNamespace(membership_logits=logits, membership_probabilities=probability,
                            seed_valid=present, module_present=present)
    model.set_native_training_epoch(700)
    a, b = model._work_probabilities(graph, dispatch, qmask)
    assert torch.equal(a, dispatch.probabilities)
    assert torch.equal(b, probability)
    model.set_native_training_epoch(701)
    a, b = model._work_probabilities(graph, dispatch, qmask)
    torch.testing.assert_close(a[:, :1], torch.softmax(d[:, :1]/.5, -1))
    torch.testing.assert_close(b, torch.where(eye, torch.ones_like(logits), torch.sigmoid(logits/.5)))
    assert torch.count_nonzero(a[:, 1:]) == 0
