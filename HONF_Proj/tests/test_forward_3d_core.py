from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.organizer import HypergraphOrganizerCore


def _classic_config() -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        field_dim=3,
        spatial_dim=3,
        coordinate_scale=[50.0, 38.0, 6.25],
        module_radius=0.5,
        num_hyperedges=6,
        hidden_dim=24,
        dropout=0.0,
        geometry_mode="nonperiodic",
        boundary_feature_mode="none",
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=24,
        pairwise_aggregation_mode="fused_query_module",
    )


def _batch(*, env_weights: torch.Tensor | None = None, seed: int = 31) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    batch_size, module_count, env_count, query_count = 2, 4, 12, 16
    extent = torch.tensor([18.0, 11.0, 4.0])
    return BatchData(
        module_centers=torch.rand(batch_size, module_count, 3, generator=generator) * extent,
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        module_features=torch.randn(batch_size, module_count, 2, generator=generator),
        global_context=torch.randn(batch_size, 11, generator=generator),
        query_xy=torch.rand(batch_size, query_count, 3, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(batch_size, query_count, 3, generator=generator),
        case_name="3d-core-test",
        metadata={},
        env_coords=torch.rand(batch_size, env_count, 3, generator=generator) * extent,
        env_features=torch.randn(batch_size, env_count, 7, generator=generator),
        query_features=torch.randn(batch_size, query_count, 7, generator=generator),
        env_weights=env_weights,
    )


def test_3d_config_is_explicit_and_d2_serialization_stays_legacy() -> None:
    config = _classic_config()
    assert config.to_dict()["spatial_dim"] == 3
    assert "spatial_dim" not in UnifiedForwardConfig().to_dict()
    with pytest.raises(ValueError, match="coordinate_scale"):
        UnifiedForwardConfig(spatial_dim=3)
    with pytest.raises(ValueError, match="periodic_axes"):
        replace(config, periodic_axes=[0])


def test_classic_3d_forward_backward_update_and_vertical_sensitivity() -> None:
    torch.manual_seed(7)
    model = HONFNeuralField(_classic_config())
    batch = _batch(env_weights=torch.rand(2, 12) + 0.1)
    padded_centers = batch.module_centers.clone()
    padded_features = batch.module_features.clone()
    padded_centers[0, 3] = float("nan")
    padded_features[0, 3] = float("nan")
    batch = replace(batch, module_centers=padded_centers, module_features=padded_features)
    with torch.no_grad():
        model(batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    query = batch.query_xy.clone().requires_grad_(True)
    probe = replace(batch, query_xy=query)
    output = model(probe)
    assert output["pred_field"].shape == (2, 16, 3)
    assert output["hyper_source_coords"].shape[-1] == 3
    assert output["mechanism_geometry_features"].shape[-1] == 13
    assert output["A_me"].shape == (2, 4, 12)

    loss = torch.nn.functional.mse_loss(output["pred_field"], batch.target_field)
    loss.backward()
    assert torch.isfinite(loss)
    assert query.grad is not None
    assert float(query.grad[..., 2].abs().sum()) > 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_classic_weighted_environment_duplication_preserves_case_measure() -> None:
    torch.manual_seed(13)
    config = _classic_config()
    organizer = HypergraphOrganizerCore(config).eval()
    batch_size, module_count, env_count, hidden = 2, 3, 5, 24
    modules = torch.randn(batch_size, module_count, hidden)
    environments = torch.randn(batch_size, env_count, hidden)
    centers = torch.rand(batch_size, module_count, 3)
    coordinates = torch.rand(batch_size, env_count, 3)
    present = torch.ones(batch_size, module_count)
    weights = torch.rand(batch_size, env_count) + 0.2
    with torch.no_grad():
        reference = organizer(
            modules,
            environments,
            centers,
            coordinates,
            present,
            env_weights=weights,
        )
        duplicated = organizer(
            modules,
            torch.repeat_interleave(environments, 2, dim=1),
            centers,
            torch.repeat_interleave(coordinates, 2, dim=1),
            present,
            env_weights=torch.repeat_interleave(weights / 2.0, 2, dim=1),
        )
        rescaled = organizer(
            modules,
            environments,
            centers,
            coordinates,
            present,
            env_weights=weights * 1.0e-20,
        )
    for key in ("hyper_state", "hyper_region_coords", "hyper_region_scale", "hyper_env_mass"):
        torch.testing.assert_close(reference[key], duplicated[key], rtol=2.0e-5, atol=2.0e-6)
        torch.testing.assert_close(reference[key], rescaled[key], rtol=2.0e-5, atol=2.0e-6)


def test_dense_3d_uses_supplied_weights_and_rejects_bad_measure() -> None:
    config = UnifiedForwardConfig(
        field_dim=3,
        spatial_dim=3,
        coordinate_scale=[50.0, 38.0, 6.25],
        hidden_dim=24,
        dropout=0.0,
        geometry_mode="nonperiodic",
        boundary_feature_mode="none",
        forward_architecture="dense_pairwise_field",
        interface_model=InterfaceFieldConfig(
            message_hidden_dim=12,
            attention_heads=4,
            coarse_latent_count=4,
            coarse_blocks=1,
            relative_fourier_frequencies=2,
            receiver_chunk_size=8,
        ),
    )
    model = InterfaceFieldCore(config)
    batch = _batch(env_weights=torch.rand(2, 12) + 0.1, seed=42)
    padded_centers = batch.module_centers.clone()
    padded_features = batch.module_features.clone()
    padded_centers[0, 3] = float("nan")
    padded_features[0, 3] = float("nan")
    batch = replace(batch, module_centers=padded_centers, module_features=padded_features)
    encoded = model.encode_case(batch)
    assert torch.isfinite(encoded.module_tokens).all()
    assert torch.equal(encoded.env_weights, batch.env_weights)
    prepared = model.prepare(encoded, encoded.module_tokens)
    query = batch.query_xy.clone().requires_grad_(True)
    output = model.decode_queries(prepared, query, batch.query_features)["pred_field"]
    loss = torch.nn.functional.mse_loss(output, batch.target_field)
    loss.backward()
    assert output.shape == batch.target_field.shape
    assert torch.isfinite(loss)
    assert query.grad is not None
    assert float(query.grad[..., 2].abs().sum()) > 0.0

    bad = replace(batch, env_weights=torch.ones(2, 11))
    with pytest.raises(ValueError, match="env_weights"):
        model.encode_case(bad)
