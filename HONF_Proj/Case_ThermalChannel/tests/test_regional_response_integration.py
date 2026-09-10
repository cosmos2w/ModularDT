"""Focused integration checks for the regional-response adapter path."""

from __future__ import annotations

import pytest
import torch

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.environment import ChannelThermalEnvironmentBuilder
from channelthermal.model import ChannelThermalHONFModel
from honf_runtime.config_loader import load_config_bundle


def test_region_ids_follow_physical_coordinate_ranks_and_duplicates() -> None:
    builder = ChannelThermalEnvironmentBuilder()
    environment = builder(
        batch_size=1,
        num_env_tokens_x=24,
        num_env_tokens_y=8,
        domain_length_x=12.0,
        domain_length_y=4.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        response_region_block_shape=(2, 2),
    )
    coordinates = environment.env_coords[0]
    ids = environment.env_region_ids[0]
    assert torch.unique(ids).numel() == 48

    permutation = torch.randperm(coordinates.shape[0], generator=torch.Generator().manual_seed(17))
    permuted_ids = builder.region_ids_from_coordinates(
        coordinates[permutation],
        num_env_tokens_x=24,
        num_env_tokens_y=8,
        block_shape=(2, 2),
    )
    torch.testing.assert_close(permuted_ids, ids[permutation])

    duplicated = torch.cat([coordinates, coordinates[[11, 11, 173]]], dim=0)
    duplicated_ids = builder.region_ids_from_coordinates(
        duplicated,
        num_env_tokens_x=24,
        num_env_tokens_y=8,
        block_shape=(2, 2),
    )
    torch.testing.assert_close(duplicated_ids[-3:], ids[[11, 11, 173]])


def test_region_ids_require_complete_rectangular_grid_metadata() -> None:
    builder = ChannelThermalEnvironmentBuilder()
    coordinates = torch.tensor(
        [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]], dtype=torch.float32
    )
    with pytest.raises(ValueError, match="rectangular grid metadata"):
        builder.region_ids_from_coordinates(
            coordinates,
            num_env_tokens_x=3,
            num_env_tokens_y=2,
            block_shape=(2, 2),
        )


def test_regional_profile_resolves_and_preserves_dense_training_policy() -> None:
    bundle = load_config_bundle("project://src/config_core/forward/regional_response_interface_context.json")
    core = bundle.core["model"]["core_honf"]
    assert core["forward_architecture"] == "regional_response_honf"
    assert core["interface_model"]["response_region_block_shape"] == [2, 2]
    assert bundle.core["training"]["seed"] == 0
    assert bundle.core["training"]["learning_rate"] == pytest.approx(3.0e-4)
    assert bundle.core["checkpointing"]["save_epoch_milestones"] == [10, 50, 100, 250, 500, 1000, 2500, 5000]

    regional_config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": "regional_response_honf",
                "interface_model": {"response_region_block_shape": [2, 2]},
                "field_dim": 5,
                "hidden_dim": 16,
            }
        }
    )
    assert regional_config.core_honf.to_dict()["interface_model"]["response_region_block_shape"] == [2, 2]
    dense_config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": "dense_pairwise_field",
                "interface_model": {},
                "field_dim": 5,
                "hidden_dim": 16,
            }
        }
    )
    assert "response_region_block_shape" not in dense_config.core_honf.to_dict()["interface_model"]


def test_regional_model_reads_shared_states_with_fine_coarse_environment() -> None:
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": "regional_response_honf",
                "interface_model": {
                    "message_hidden_dim": 24,
                    "attention_heads": 4,
                    "coarse_latent_count": 4,
                    "coarse_blocks": 1,
                    "main_latent_count": 4,
                    "main_latent_blocks": 2,
                    "local_radius_factor": 2.5,
                    "relative_fourier_frequencies": 2,
                    "receiver_chunk_size": 3,
                    "activation_checkpointing": False,
                    "response_region_block_shape": [2, 2],
                },
                "hidden_dim": 32,
                "field_dim": 5,
                "domain_length_x": 4.0,
                "domain_length_y": 2.0,
                "coordinate_scale": [4.0, 2.0],
                "module_radius": 0.2,
                "num_env_tokens_x": 4,
                "num_env_tokens_y": 4,
                "dropout": 0.0,
                "boundary_feature_mode": "none",
                "position_fourier_frequencies": 2,
                "query_fourier_frequencies": 2,
            },
            "channelthermal": {
                "use_local_surrogate": False,
                "internal_prediction_mode": "global_head",
                "default_num_interface_points": 8,
            },
        }
    )
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False).eval()
    output = model(
        query_xy=torch.tensor([[[0.25, 0.25], [1.5, 0.75], [3.5, 1.75]]]),
        re=torch.tensor([[100.0]]),
        u_in=torch.tensor([[1.0]]),
        module_centers=torch.tensor([[[1.0, 0.75], [3.0, 1.25]]]),
        heat_powers=torch.tensor([[1.0, 2.0]]),
        module_present=torch.tensor([[1.0, 1.0]]),
        material_params=torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.2]]),
        return_prepared_state=True,
        return_routing_maps=True,
    )
    assert torch.isfinite(output["pred_field"]).all()
    prepared = output["prepared_state"].prepared
    assert prepared.encoded.env_region_ids is not None
    assert prepared.encoded.env_region_ids.shape == (1, 16)
    assert prepared.backend_state["regional_response_states"].shape[1] == 4
    assert output["interaction_aux"]["regional_response_count"].item() == 4
    assert torch.isfinite(output["interaction_aux"]["dense_module_context_norm"]).all()
    assert torch.isfinite(output["interaction_aux"]["regional_context_norm"]).all()
    assert output["routing_aux"]["regional_environment_attention"].shape[-1] == 4
