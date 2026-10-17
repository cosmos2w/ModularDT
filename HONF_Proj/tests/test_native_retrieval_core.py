from dataclasses import replace

import pytest
import torch

from honf_forward_core.native_retrieval.core import NativeRetrievalCore
from tests.test_content_hypergraph_v2_core import _core, _query, _scene


def test_all_open_parent_recovery_gradients_loss_and_optimizer():
    parent = _core("A2-D").double().train()
    child = NativeRetrievalCore.from_a2d(parent)
    scene = _scene(1)
    scene = replace(scene, **{name: value.double() for name, value in vars(scene).items()
                              if isinstance(value, torch.Tensor) and value.is_floating_point()})
    query = _query(1)
    query = replace(query, coordinates=query.coordinates.double(), features=query.features.double())
    heat = torch.tensor([[1., -.2, .5]], dtype=torch.double)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3) for model in (parent, child)]
    losses = []
    for model, optimizer in zip((parent, child), optimizers):
        x = replace(query, coordinates=query.coordinates.detach().clone().requires_grad_())
        prepared = model.prepare(scene)
        output = model.decode(prepared, x, heat)
        loss = output.nonlinear_field.square().sum() + output.thermal_temperature.square().sum()
        loss.backward()
        losses.append((loss.detach(), x.coordinates.grad.clone()))
    assert torch.equal(losses[0][0], losses[1][0])
    assert torch.equal(losses[0][1], losses[1][1])
    for (name, left), (other, right) in zip(parent.named_parameters(), child.named_parameters()):
        assert name == other
        if left.grad is not None:
            assert torch.equal(left.grad, right.grad), name
    for optimizer in optimizers:
        optimizer.step()
    for name, value in parent.state_dict().items():
        assert torch.equal(value, child.state_dict()[name]), name


def test_policy_curriculum_roundtrip_and_preparation_lifetime():
    model = NativeRetrievalCore.from_a2d(_core("A2-D"))
    assert model.native_routing_state()["strength"] == 0
    model.set_native_training_epoch(1000)
    assert model.native_routing_state()["strength"] == 1
    restored = NativeRetrievalCore.from_config(model.get_config())
    restored.load_state_dict(model.state_dict())
    assert restored.native_routing_state() == model.native_routing_state()
    scene, query = _scene(1), _query(1)
    prepared = model.prepare(scene)
    assert prepared.native_hypergraph.mm_snapshot.stage_identity == "cycle1/mm"
    assert prepared.native_hypergraph.me_snapshot.stage_identity == "cycle1/me"
    output = model.decode(prepared, query, torch.ones(1, 3))
    (output.nonlinear_field.sum() + output.thermal_temperature.sum()).backward()
    with pytest.raises(ValueError, match="consumed"):
        model.decode(prepared, query)
    model.set_native_training_epoch(1008)
    assert model.native_routing_state()["exploration"]
    assert not model._uses_native_retrieval()
    model.set_native_training_epoch(2008)
    assert not model.native_routing_state()["exploration"]
    assert model._uses_native_retrieval()


def test_native_geometry_heat_laws_and_rebuilt_query_chunks():
    parent = _core("A2-D").double().eval()
    model = NativeRetrievalCore.from_a2d(parent)
    model.set_native_training_epoch(1000)
    scene = _scene(1)
    scene = replace(scene, **{name: value.double() for name, value in vars(scene).items()
                              if isinstance(value, torch.Tensor) and value.is_floating_point()})
    query = _query(1)
    query = replace(query, coordinates=query.coordinates.double(), features=query.features.double())
    with torch.no_grad():
        prepared = model.prepare(scene)
        heat = torch.tensor([[1., -.3, .7]], dtype=torch.double)
        out = model.decode(prepared, query, heat, accumulation_dtype=torch.float64)
        zero = model.decode(prepared, query, heat * 0, accumulation_dtype=torch.float64)
        twice = model.decode(prepared, query, 2 * heat, accumulation_dtype=torch.float64)
        assert torch.count_nonzero(zero.thermal_temperature) == 0
        assert torch.equal(out.nonlinear_field, twice.nonlinear_field)
        torch.testing.assert_close(twice.thermal_temperature, 2 * out.thermal_temperature, atol=1e-12, rtol=1e-12)
        blocks = []
        for start, stop in ((0, 2), (2, 5)):
            q = replace(query, coordinates=query.coordinates[:, start:stop], features=query.features[:, start:stop],
                        query_ids=query.query_ids[:, start:stop])
            blocks.append(model.decode(prepared, q, heat).thermal_temperature)
        torch.testing.assert_close(torch.cat(blocks, 1), out.thermal_temperature, atol=1e-12, rtol=1e-12)
        changed = replace(scene, module_measure=scene.module_measure.clone())
        changed.module_measure[0, 0] *= 2
        prepared.scene.module_measure[0, 0] *= 1.1
        with pytest.raises(ValueError, match="changed|stale"):
            model.decode(prepared, query)


def test_core_rejects_wrong_parents_and_uncontrolled_environment_consumer():
    with pytest.raises(ValueError, match="uncompressed"):
        NativeRetrievalCore(_core("A2-H6").get_config())
    with pytest.raises(ValueError, match="pilot"):
        NativeRetrievalCore(_core("A2-D").get_config(), environment_value_reader=True)
