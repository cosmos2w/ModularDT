"""Closed C1 support must preserve dense zero-gradient optimizer semantics."""
import copy
from dataclasses import replace

import pytest
import torch

from tests.test_continuous_joint import _model, _double
from tests.test_content_hypergraph_v2_core import _scene, _query


@pytest.mark.parametrize('thermal_loss', [False, True])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_empty_support_keeps_zero_gradients_and_adamw_state(thermal_loss, checkpoint):
    _, packed = _model('direct', checkpoint=checkpoint, tile=10)
    packed.train()
    with torch.no_grad():
        packed.router.direct_parameters.module.bias.fill_(-100.)
        packed.router.direct_parameters.environment.bias.fill_(-100.)
    dense = copy.deepcopy(packed)
    dense.native_backend = 'dense'
    scene, query = _double(_scene(1)), _double(_query(1))
    # Fixed ordinary inventory, outside the compact local support. Every
    # optional C1 gate is closed, so its composed first derivative is zero.
    query = replace(query, coordinates=torch.full_like(query.coordinates, 100.))
    heat = torch.ones(1, 3, dtype=torch.double)
    gradients, states = [], []
    for model in (packed, dense):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        result = model.decode(model.prepare(scene), query, heat)
        loss = result.nonlinear_field.square().sum()
        if thermal_loss:
            loss = loss + result.thermal_temperature.square().sum()
        loss.backward()
        gradients.append({name: None if p.grad is None else p.grad.clone()
                          for name, p in model.named_parameters()})
        optimizer.step()
        states.append({name: copy.deepcopy(optimizer.state.get(p))
                       for name, p in model.named_parameters()})
        if model is packed:
            assert result.work_counts['joint_M_value_rows'] == 0
            assert result.work_counts['joint_E_value_rows'] == 0
    for name in gradients[0]:
        left, right = gradients[0][name], gradients[1][name]
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        left_state, right_state = states[0][name], states[1][name]
        assert (left_state is None) == (right_state is None), name
        if left_state is not None:
            for key in left_state:
                torch.testing.assert_close(left_state[key], right_state[key], rtol=0, atol=0)
    for (name, left), (_, right) in zip(packed.named_parameters(), dense.named_parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
