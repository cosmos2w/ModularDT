"""Exact opt-in one-group execution, including biased projections and VJPs."""

import copy
from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded():
    generator = torch.Generator().manual_seed(731)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(2, 4, 8, generator=generator),
        env_tokens=torch.randn(2, 7, 8, generator=generator),
        global_token=torch.randn(2, 8, generator=generator),
        module_centers=torch.rand(2, 4, 2, generator=generator),
        env_coords=torch.rand(2, 7, 2, generator=generator),
        module_present=torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]]),
        module_features=torch.randn(2, 4, 3, generator=generator), env_features=None,
        env_weights=torch.tensor([[1., .5, 2., 1., 3., .75, 1.]]).expand(2, -1),
        coordinate_scale=torch.ones(1, 1, 2))


@pytest.mark.parametrize("modules,environments", [(1, 0), (2, 3), (4, 7)])
@pytest.mark.parametrize("execution", ["dense_masked_reference", "rectangular_subset"])
def test_nonzero_global_fast_outputs_gradients_and_adam_step(modules, environments, execution):
    torch.manual_seed(319)
    base = TypedHypergraphField(8, 12, 2, 2,
        architecture="native_context_global_control_honf", spatial_dim=2,
        module_characteristic_length=.1, options={"organizer_dim": 16}).double().eval()
    encoded = _encoded()
    encoded = replace(encoded, module_tokens=encoded.module_tokens[:, :modules].double(),
        module_centers=encoded.module_centers[:, :modules].double(),
        module_present=encoded.module_present[:, :modules].double(),
        env_tokens=encoded.env_tokens[:, :environments].double(),
        env_coords=encoded.env_coords[:, :environments].double(),
        env_weights=encoded.env_weights[:, :environments].double(),
        global_token=encoded.global_token.double(), coordinate_scale=encoded.coordinate_scale.double())
    query = torch.rand(2, 5, 2, dtype=torch.double)
    features = torch.rand(2, 5, 6, dtype=torch.double)
    base.read(base.prepare(encoded, encoded.module_tokens), encoded, query, features)
    with torch.no_grad():
        for head in base.organizer.control_heads.values():
            head[-1].weight.normal_(std=.1)
        for head in base.control_gain.values():
            head.weight.normal_(std=.1)
            head.bias.uniform_(-.1, .1)
        # Even a nonzero constant QE score must cancel.
        base.control_score.weight.normal_(std=.2)
        base.control_score.bias.fill_(.17)
    fast = copy.deepcopy(base)
    fast.global_fast_reader = True
    assert base.state_dict().keys() == fast.state_dict().keys()
    values, gradients = [], []
    for model in (base, fast):
        model.set_execution_mode(execution, receiver_chunk_size=2)
        q = query.clone().requires_grad_()
        tokens = encoded.module_tokens.clone().requires_grad_()
        record = replace(encoded, module_tokens=tokens)
        output, _ = model.read(model.prepare(record, tokens), record, q, features)
        output.square().mean().backward()
        values.append(output)
        gradients.append((q.grad, tokens.grad))
    torch.testing.assert_close(values[1], values[0], atol=1e-12, rtol=1e-10)
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    for (name, parent), (_, child) in zip(base.named_parameters(), fast.named_parameters()):
        assert (parent.grad is None) == (child.grad is None), name
        if parent.grad is not None:
            torch.testing.assert_close(child.grad, parent.grad, atol=1e-12, rtol=1e-10)
    for model in (base, fast):
        torch.optim.Adam(model.parameters(), lr=.001).step()
    for name, value in base.state_dict().items():
        if isinstance(value, torch.nn.parameter.UninitializedParameter):
            assert isinstance(fast.state_dict()[name], torch.nn.parameter.UninitializedParameter)
            continue
        torch.testing.assert_close(fast.state_dict()[name], value, atol=1e-12, rtol=1e-10)


def test_fast_numerical_actions_have_no_pair_control_axis():
    model = TypedHypergraphField(8, 12, 2, 2,
        architecture="native_context_global_control_honf", spatial_dim=2,
        module_characteristic_length=.1, options={"organizer_dim": 16, "global_fast_reader": True})
    encoded = _encoded()
    prepared = model.prepare(encoded, encoded.module_tokens)
    for tau, action in prepared["hypergraph_actions"].items():
        assert action.shape[:2] == (2, 1)
        assert prepared["hypergraph_accesses"].get(tau) is None or prepared["hypergraph_accesses"][tau].projected.shape[1:3] == (1, 1)
    # Semantic diagnostic export retains the historical full pair view.
    export = model.export_typed_state(prepared)
    access = export["receiver_access"](encoded.module_centers, "MM")
    assert access.control.shape == (2, 4, 4, 16)
