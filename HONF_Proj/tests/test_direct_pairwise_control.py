"""Pair-F identity, trained content paths, measures and true full-access work."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.direct_pairwise_control import DirectPairwiseControlField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded(dimension=2):
    generator = torch.Generator().manual_seed(731)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(2, 4, 8, generator=generator),
        env_tokens=torch.randn(2, 7, 8, generator=generator),
        global_token=torch.randn(2, 8, generator=generator),
        module_centers=torch.rand(2, 4, dimension, generator=generator),
        env_coords=torch.rand(2, 7, dimension, generator=generator),
        module_present=torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]]),
        module_features=torch.randn(2, 4, 3, generator=generator), env_features=None,
        env_weights=torch.tensor([[1., .5, 2., 1., 3., .75, 1.]]).expand(2, -1),
        coordinate_scale=torch.ones(1, 1, dimension))


@pytest.mark.parametrize("dimension", [2, 3])
def test_identity_matches_b_fine_and_control_heads_receive_training_signal(dimension):
    torch.manual_seed(0)
    encoded = _encoded(dimension)
    query, features = torch.rand(2, 5, dimension), torch.rand(2, 5, 6)
    fine = DensePairwiseField(8, 12, 2, 2).eval()
    pair = DirectPairwiseControlField(8, 12, 2, 2, control_hidden_dim=12).eval()
    expected, _ = fine.read(fine.prepare(encoded, encoded.module_tokens), encoded, query, features)
    pair.read(pair.prepare(encoded, encoded.module_tokens), encoded, query, features)
    pair.load_state_dict({**pair.state_dict(), **fine.state_dict()})
    actual, ledger = pair.read(pair.prepare(encoded, encoded.module_tokens), encoded, query, features)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert ledger["pair_QM_executed_rows"].item() == 2 * 5 * 4
    assert ledger["pair_QE_executed_rows"].item() == 2 * 5 * 7
    assert ledger["pair_QM_control_executed_rows"].item() == 2 * 5 * 4
    assert ledger["pair_QE_control_calls"].item() == 1
    optimizer = torch.optim.Adam(pair.parameters(), lr=.01)
    targets = torch.randn_like(actual)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        actual, _ = pair.read(pair.prepare(encoded, encoded.module_tokens), encoded, query, features)
        (actual - targets).square().mean().backward()
        for head in pair.pair_controls.values():
            assert head.net[-1].weight.grad is not None
            assert head.net[-1].weight.grad.abs().sum() > 0
            if step > 0:
                assert head.net[0].weight.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.allclose(actual, expected)


def test_trained_pair_refinement_permutation_and_first_gradient_pullback():
    torch.manual_seed(17)
    original = _encoded()
    encoded = replace(original, **{name: value.double() for name, value in vars(original).items()
        if torch.is_tensor(value) and value.is_floating_point()})
    query, features = torch.rand(2, 5, 2, dtype=torch.float64), torch.rand(2, 5, 6, dtype=torch.float64)
    pair = DirectPairwiseControlField(8, 12, 2, 2, control_hidden_dim=12).double().eval()
    pair.read(pair.prepare(encoded, encoded.module_tokens), encoded, query, features)
    with torch.no_grad():
        for head in pair.pair_controls.values():
            head.net[-1].weight.normal_(0, .1)
    # Refine atom1 into unequal children, retain all content and its length.
    indices = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6])
    coefficients = torch.tensor([1., .3, .7, 1., 1., 1., 1., 1.], dtype=torch.float64)
    child_tokens = encoded.env_tokens.index_select(1, indices).clone().requires_grad_()
    parent_tokens = encoded.env_tokens.clone().requires_grad_()
    parent = replace(encoded, env_tokens=parent_tokens)
    refined = replace(encoded, env_tokens=child_tokens,
        env_coords=encoded.env_coords.index_select(1, indices),
        env_weights=encoded.env_weights.index_select(1, indices) * coefficients)
    a, _ = pair.read(pair.prepare(parent, parent.module_tokens), parent, query, features)
    b, _ = pair.read(pair.prepare(refined, refined.module_tokens), refined, query, features)
    torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-11)
    a.square().sum().backward()
    pair.zero_grad(set_to_none=True)
    b.square().sum().backward()
    pullback = torch.zeros_like(parent_tokens).index_add_(1, indices, child_tokens.grad)
    torch.testing.assert_close(pullback, parent_tokens.grad, rtol=1e-9, atol=1e-10)
    permutation = torch.tensor([6, 4, 0, 2, 1, 5, 3])
    permuted = replace(encoded, env_tokens=encoded.env_tokens.index_select(1, permutation),
        env_coords=encoded.env_coords.index_select(1, permutation),
        env_weights=encoded.env_weights.index_select(1, permutation))
    c, _ = pair.read(pair.prepare(permuted, permuted.module_tokens), permuted, query, features)
    torch.testing.assert_close(c, a, rtol=1e-10, atol=1e-11)


def test_controls_use_individual_source_content_without_collective_pooling():
    pair = DirectPairwiseControlField(8, 12, 2, 2, control_hidden_dim=12).eval()
    encoded = _encoded()
    control = pair._control("MM", encoded.module_tokens, encoded.module_tokens,
        encoded.module_centers, encoded.module_centers, encoded)
    with torch.no_grad():
        pair.pair_controls["MM"].net[-1].weight.normal_(0, .1)
    control = pair._control("MM", encoded.module_tokens, encoded.module_tokens,
        encoded.module_centers, encoded.module_centers, encoded)
    changed = encoded.module_tokens.clone()
    changed[:, 1] += 3
    perturbed = pair._control("MM", encoded.module_tokens, changed,
        encoded.module_centers, encoded.module_centers, encoded)
    torch.testing.assert_close(perturbed[:, :, [0, 2, 3]], control[:, :, [0, 2, 3]], rtol=0, atol=0)
    assert (perturbed[:, :, 1] - control[:, :, 1]).abs().max() > 1e-5
    assert not hasattr(pair, "organizer")


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("architecture", ["direct_pairwise_control_honf", "faithful_receiver_hypergraph_honf"])
def test_new_registered_backends_chunk_backward_and_capacity_scope(dimension, architecture):
    encoded = _encoded(dimension)
    config = UnifiedForwardConfig(spatial_dim=dimension, coordinate_scale=[1.] * dimension,
        boundary_feature_mode="none", periodic_axes=[], field_dim=3, hidden_dim=8,
        forward_architecture=architecture,
        interface_model=InterfaceFieldConfig(message_hidden_dim=12, attention_heads=2,
            relative_fourier_frequencies=2, receiver_chunk_size=3))
    core = InterfaceFieldCore(config).eval()
    batch = BatchData(module_centers=encoded.module_centers, module_present=encoded.module_present,
        module_features=encoded.module_features, global_context=encoded.global_token,
        query_xy=torch.rand(2, 11, dimension), target_field=None, query_time=None,
        case_name="faithfulness-test", metadata={}, env_coords=encoded.env_coords,
        env_features=torch.randn(2, 7, 3), env_weights=encoded.env_weights)
    record = core.encode_case(batch)
    prepared = core.prepare(record, record.module_tokens)
    output = core.decode_queries(prepared, batch.query_xy, receiver_chunk_size=3, return_routing_maps=True)
    complete = core.decode_queries(prepared, batch.query_xy, receiver_chunk_size=30, return_routing_maps=True)
    torch.testing.assert_close(output["pred_field"], complete["pred_field"])
    output["pred_field"].square().mean().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in core.parameters() if parameter.grad is not None)
    if architecture == "direct_pairwise_control_honf":
        assert output["pair_QM_executed_rows"].item() == 2 * 11 * 4
        assert output["pair_QE_control_executed_rows"].item() == 2 * 11 * 7
        assert output["pair_QE_control_calls"].item() == 4
        assert complete["pair_QE_control_calls"].item() == 1
    else:
        assert core.backend.organizer.faithful_controls
        assert core.backend.organizer.measure_consistent
