from __future__ import annotations

import torch

from honf_forward_core.interface_fields.common import BiasedMultiheadAttention
from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.regional_response import (
    RegionalResponseField,
    pool_region_weighted,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded(*, batch: int = 1, modules: int = 3, environment: int = 6, hidden: int = 12) -> EncodedInterfaceCase:
    generator = torch.Generator().manual_seed(91)
    module_centers = torch.rand(batch, modules, 2, generator=generator)
    env_coords = torch.rand(batch, environment, 2, generator=generator)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator),
        env_tokens=torch.randn(batch, environment, hidden, generator=generator),
        global_token=torch.randn(batch, hidden, generator=generator),
        module_centers=module_centers,
        env_coords=env_coords,
        module_present=torch.tensor([[1.0, 1.0, 0.0]]).expand(batch, -1).clone(),
        module_features=torch.randn(batch, modules, 4, generator=generator),
        env_features=None,
        env_weights=torch.tensor([1.0, 2.0, 1.0, 3.0, 0.5, 1.5]).expand(batch, -1).clone(),
        coordinate_scale=torch.ones(1, 1, 2),
    )


def test_weighted_pool_preserves_mass_centroid_and_duplicate_split() -> None:
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0], [5.0, 7.0], [9.0, 11.0]]])
    coords = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [8.0, 8.0], [0.0, 2.0], [2.0, 2.0]]])
    region_ids = torch.tensor([[7, 7, -1, 10, 10]])
    weights = torch.tensor([[1.0, 2.0, 0.0, 1.0, 3.0]])
    pooled = pool_region_weighted(values, region_ids, weights, coords)

    assert pooled.region_ids.tolist() == [[7, 10]]
    torch.testing.assert_close(pooled.mass, torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(pooled.values, torch.tensor([[[7.0 / 3.0, 10.0 / 3.0], [8.0, 10.0]]]))
    torch.testing.assert_close(pooled.centroids, torch.tensor([[[4.0 / 3.0, 0.0], [1.5, 2.0]]]))
    assert pooled.valid.tolist() == [[True, True]]

    unsplit = pool_region_weighted(
        torch.tensor([[[1.0, 2.0], [5.0, 7.0]]]),
        torch.tensor([[7, 10]]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[[0.0, 0.0], [2.0, 2.0]]]),
    )
    split = pool_region_weighted(
        torch.tensor([[[1.0, 2.0], [1.0, 2.0], [5.0, 7.0], [5.0, 7.0]]]),
        torch.tensor([[7, 7, 10, 10]]),
        torch.tensor([[1.0, 2.0, 1.0, 3.0]]),
        torch.tensor([[[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]]]),
    )
    torch.testing.assert_close(unsplit.values, split.values)
    torch.testing.assert_close(unsplit.mass, split.mass)
    torch.testing.assert_close(unsplit.centroids, split.centroids)


def test_weighted_pool_is_invariant_to_token_permutation() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(2, 9, 4, generator=generator)
    coords = torch.randn(2, 9, 2, generator=generator)
    ids = torch.tensor([0, 3, 0, 5, 3, 5, 9, 9, -1]).expand(2, -1)
    weights = torch.rand(2, 9, generator=generator) + 0.1
    permutation = torch.tensor([6, 1, 4, 2, 0, 8, 3, 7, 5])
    reference = pool_region_weighted(values, ids, weights, coords)
    permuted = pool_region_weighted(values[:, permutation], ids[:, permutation], weights[:, permutation], coords[:, permutation])
    torch.testing.assert_close(reference.values, permuted.values)
    torch.testing.assert_close(reference.mass, permuted.mass)
    torch.testing.assert_close(reference.centroids, permuted.centroids)
    assert reference.region_ids.tolist() == permuted.region_ids.tolist()


def test_projected_attention_reproduces_constant_within_region_weighted_read() -> None:
    attention = BiasedMultiheadAttention(hidden_dim=8, num_heads=2)
    torch.nn.init.eye_(attention.output.weight)
    torch.nn.init.zeros_(attention.output.bias)
    query = torch.ones(1, 2, 1, 4)
    fine_key = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
          [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]]
    )
    fine_value = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0]],
          [[9.0, 10.0, 11.0, 12.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0], [13.0, 14.0, 15.0, 16.0]]]]
    )
    fine_weights = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
    fine_output, fine_attention = attention.read_projected(
        query, fine_key, fine_value, log_weights=torch.log(fine_weights), return_attention=True
    )
    pooled_key = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
          [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]]]
    )
    pooled_value = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
          [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]
    )
    pooled_output, pooled_attention = attention.read_projected(
        query, pooled_key, pooled_value, log_weights=torch.log(torch.tensor([[4.0, 6.0]])), return_attention=True
    )
    torch.testing.assert_close(fine_output, pooled_output)
    assert fine_attention is not None and pooled_attention is not None
    torch.testing.assert_close(fine_attention.sum(dim=-1), torch.ones(1, 2, 1))
    torch.testing.assert_close(pooled_attention.sum(dim=-1), torch.ones(1, 2, 1))


def test_singleton_native_grouping_matches_dense_forward_and_backward() -> None:
    encoded = _encoded(environment=6)
    encoded = EncodedInterfaceCase(
        **{
            field: value.double() if torch.is_tensor(value) and value.is_floating_point() else value
            for field, value in encoded.__dict__.items()
        }
    )
    module_states = torch.randn(1, 3, 12, generator=torch.Generator().manual_seed(12)).double()
    receivers = torch.rand(1, 5, 2, generator=torch.Generator().manual_seed(13)).double()
    receiver_features = torch.randn(1, 5, 12, generator=torch.Generator().manual_seed(14)).double()
    dense = DensePairwiseField(12, 8, 3, 2).double()
    regional = RegionalResponseField(12, 8, 3, 2).double()
    with torch.no_grad():
        dense.read(dense.prepare(encoded, module_states), encoded, receivers, receiver_features)
        regional.prepare(encoded, module_states, region_ids=torch.arange(6))
        regional.read(
            regional.prepare(encoded, module_states, region_ids=torch.arange(6)),
            encoded,
            receivers,
            receiver_features,
        )
    regional.load_state_dict(dense.state_dict(), strict=True)
    states_dense = module_states.detach().clone().requires_grad_()
    states_regional = module_states.detach().clone().requires_grad_()
    dense_context, _ = dense.read(dense.prepare(encoded, states_dense), encoded, receivers, receiver_features)
    regional_context, _ = regional.read(
        regional.prepare(encoded, states_regional, region_ids=torch.arange(6)),
        encoded,
        receivers,
        receiver_features,
    )
    torch.testing.assert_close(regional_context, dense_context, rtol=2.0e-5, atol=2.0e-6)
    dense_context.sum().backward()
    regional_context.sum().backward()
    assert states_dense.grad is not None and states_regional.grad is not None
    torch.testing.assert_close(states_regional.grad, states_dense.grad, rtol=2.0e-5, atol=2.0e-6)
    for (dense_name, dense_parameter), (regional_name, regional_parameter) in zip(
        dense.named_parameters(), regional.named_parameters(), strict=True
    ):
        assert dense_name == regional_name
        if dense_parameter.grad is None or regional_parameter.grad is None:
            assert dense_parameter.grad is None and regional_parameter.grad is None
        else:
            torch.testing.assert_close(
                regional_parameter.grad,
                dense_parameter.grad,
                rtol=2.0e-5,
                atol=2.0e-6,
            )


def test_native_projection_cache_is_live_refreshable_and_chunk_consistent() -> None:
    encoded = _encoded(environment=6)
    model = RegionalResponseField(12, 8, 3, 2)
    states = torch.randn(1, 3, 12, generator=torch.Generator().manual_seed(32), requires_grad=True)
    region_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    first = model.prepare(encoded, states, region_ids=region_ids)
    assert first["regional_keys"].requires_grad
    receivers = torch.rand(1, 11, 2, generator=torch.Generator().manual_seed(33))
    receiver_features = torch.randn(1, 11, 12, generator=torch.Generator().manual_seed(34))
    whole, _ = model.read(first, encoded, receivers, receiver_features)
    chunks = [
        model.read(first, encoded, receivers[:, start : start + 4], receiver_features[:, start : start + 4])[0]
        for start in range(0, 11, 4)
    ]
    torch.testing.assert_close(whole, torch.cat(chunks, dim=1), rtol=2.0e-5, atol=2.0e-6)
    whole.sum().backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    refreshed = model.prepare(encoded, states.detach() + 0.25, region_ids=region_ids)
    assert not torch.allclose(first["regional_keys"].detach(), refreshed["regional_keys"].detach())


def test_native_module_permutation_and_padding_do_not_change_the_read() -> None:
    encoded = _encoded(modules=3, environment=6)
    model = RegionalResponseField(12, 8, 3, 2).eval()
    states = torch.randn(1, 3, 12, generator=torch.Generator().manual_seed(44))
    receivers = torch.rand(1, 7, 2, generator=torch.Generator().manual_seed(45))
    receiver_features = torch.randn(1, 7, 12, generator=torch.Generator().manual_seed(46))
    region_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    reference, _ = model.read(
        model.prepare(encoded, states, region_ids=region_ids),
        encoded,
        receivers,
        receiver_features,
    )

    permutation = torch.tensor([2, 0, 1])
    permuted_encoded = EncodedInterfaceCase(
        module_tokens=encoded.module_tokens[:, permutation],
        env_tokens=encoded.env_tokens,
        global_token=encoded.global_token,
        module_centers=encoded.module_centers[:, permutation],
        env_coords=encoded.env_coords,
        module_present=encoded.module_present[:, permutation],
        module_features=encoded.module_features[:, permutation],
        env_features=encoded.env_features,
        env_weights=encoded.env_weights,
        coordinate_scale=encoded.coordinate_scale,
        env_region_ids=encoded.env_region_ids,
    )
    permuted, _ = model.read(
        model.prepare(permuted_encoded, states[:, permutation], region_ids=region_ids),
        permuted_encoded,
        receivers,
        receiver_features,
    )
    torch.testing.assert_close(reference, permuted, rtol=1.0e-5, atol=1.0e-6)

    padding = 2
    padded_encoded = EncodedInterfaceCase(
        module_tokens=torch.cat([encoded.module_tokens, encoded.module_tokens.new_zeros(1, padding, 12)], dim=1),
        env_tokens=encoded.env_tokens,
        global_token=encoded.global_token,
        module_centers=torch.cat([encoded.module_centers, encoded.module_centers.new_zeros(1, padding, 2)], dim=1),
        env_coords=encoded.env_coords,
        module_present=torch.cat([encoded.module_present, encoded.module_present.new_zeros(1, padding)], dim=1),
        module_features=torch.cat([encoded.module_features, encoded.module_features.new_zeros(1, padding, 4)], dim=1),
        env_features=encoded.env_features,
        env_weights=encoded.env_weights,
        coordinate_scale=encoded.coordinate_scale,
        env_region_ids=encoded.env_region_ids,
    )
    padded, _ = model.read(
        model.prepare(padded_encoded, torch.cat([states, states.new_zeros(1, padding, 12)], dim=1), region_ids=region_ids),
        padded_encoded,
        receivers,
        receiver_features,
    )
    torch.testing.assert_close(reference, padded, rtol=1.0e-5, atol=1.0e-6)
