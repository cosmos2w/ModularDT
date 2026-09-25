"""The static profile follows the inherited physical training path."""

from __future__ import annotations

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training.optimizer import build_forward_optimizer


def test_source_conditioned_model_uses_default_optimizer_and_has_no_detail_group() -> None:
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": "source_conditioned_pairwise_honf",
                "interface_model": {
                    "message_hidden_dim": 8,
                    "attention_heads": 2,
                    "relative_fourier_frequencies": 2,
                    "receiver_chunk_size": 8,
                    "activation_checkpointing": False,
                    "group_count": 12,
                    "source_normalizer": "entmax15",
                    "group_control_dim": 16,
                    "environment_refinement_normalizer": "sparsemax",
                },
                "field_dim": 5,
                "hidden_dim": 16,
                "domain_length_x": 12.0,
                "domain_length_y": 6.0,
                "coordinate_scale": [12.0, 6.0],
                "module_radius": 0.45,
                "num_env_tokens_x": 4,
                "num_env_tokens_y": 3,
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
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False)
    optimizer, inventory = build_forward_optimizer(
        model,
        {
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-5,
            "organizer_learning_rate": None,
        },
    )

    assert optimizer.defaults["lr"] == 3.0e-4
    assert inventory["mode"] == "single"
    assert [group["name"] for group in inventory["groups"]] == ["all"]
    parameter_names = inventory["groups"][0]["parameter_names"]
    assert parameter_names
    assert not any("functional_detail_controller" in name for name in parameter_names)
    assert not any("query_projection" in name for name in parameter_names)
    assert not any("query_group_projection" in name for name in parameter_names)
